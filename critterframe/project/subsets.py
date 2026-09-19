"""
Create named, persisted selections of occurrences.

One project can need several recipes -- specimens from three museums shot to
three standards need three crop regions to reach the same wings. Define the
groups once, then point a different recipe at each.

Definitions live in definitions/subsets.toml, hand-editable by design. A subset
selects rows and never copies them, so an occurrence can belong to several, and
running over one leaves every other subset's masks and metrics untouched.

Every subset also gets `created_at` and, where given, a free-text `note` --
provenance a person reads later, never resolved or replayed, the same
manifest pattern imports, exports, and model registration already use
elsewhere in the package. An `occurrence_ids` subset additionally stores its
own resolved `{count, ids_hash}`, since unlike a `column`/`query` rule its
membership is frozen and that digest can never go stale.
"""

import logging
from datetime import datetime, timezone

from .. import selectionhelpers
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, ids_record, load_occurrences
from ..storage.jsonfiles import atomic_write
from . import paths

logger = logging.getLogger(__name__)

try:                                    # 3.11+
    import tomllib
except ModuleNotFoundError:             # 3.10, via the tomli backport
    import tomli as tomllib


def _toml_value(value):
    """Serialize one scalar/list value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dump_toml(subsets):
    """
    Write the subset table as TOML by hand.

    The structure here is fixed and shallow -- a table per subset holding
    strings, lists of strings, and nothing nested -- so a dozen lines of
    serializer beats taking on a TOML-writing dependency for it. Reading uses
    the standard library's tomllib, which is where correctness actually
    matters, since that's what has to cope with whatever a human hand-edits.
    """
    lines = [
        "# CritterFrame subsets: named selections of occurrences, each",
        "# intended to receive its own processing recipe. Hand-editable.",
        "",
    ]
    for name, definition in sorted(subsets.items()):
        lines.append(f"[subsets.{name}]")
        for key, value in definition.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def load_subsets(project_path):
    """
    Every subset definition, as {name: definition}. Empty if the project has no
    subsets.toml, which is the normal case rather than an error.
    """
    subsets_path = paths.subsets_path(project_path)
    if not subsets_path.exists():
        return {}

    with subsets_path.open("rb") as handle:
        return tomllib.load(handle).get("subsets", {})


def _save_subsets(project_path, subsets):
    """
    Write the whole subset table, replacing whatever was there.

    Atomically and as UTF-8: every subset a project has lives in this one
    file, and tomllib reads UTF-8, so a note or a value with an accent in it
    written in the platform's own encoding comes back unreadable.
    """
    subsets_path = paths.subsets_path(project_path)
    with atomic_write(subsets_path) as handle:
        handle.write(_dump_toml(subsets))

    logger.info("wrote %d subset definition(s) -> %s", len(subsets), subsets_path)
    return subsets


def define_subset(project_path, name, column=None, values=None, query=None,
                  occurrence_ids=None, from_subset=None, note=None):
    """
    Define (or redefine) one subset. Exactly one selection rule must be given.

    - `name` -- what to call it, e.g. `"amnh"`.
    - `column`, `values` -- select occurrences whose `column` is one of
      `values`, e.g. a collection, a device, a source.
    - `query` -- a pandas query against the occurrence table, e.g.
      `"year >= 2020 and country == 'Panama'"`.
    - `occurrence_ids` -- an explicit list, for a hand-picked selection with
      no rule behind it.
    - `from_subset` -- freeze another (possibly live) subset's CURRENT
      membership under this new name. Shorthand for `occurrence_ids=
      select_ids(project_path, subset=from_subset)` -- the new subset is an
      independent snapshot afterward, so redefining `from_subset` later, or
      the project growing under it, never moves this one. For locking in a
      `column`/`query` subset's matches as a stable id list something else
      (a training split, say) can depend on.
    - `note` -- optional free text on why this subset exists, e.g. "held-out
      test set, reviewed by hand 2026-03". Never read back by anything --
      purely for a human reading subsets.toml later, the same way
      records.models.register_model's `notes` works. Most worth giving to an
      `occurrence_ids` subset, since unlike `column`/`query` the rule itself
      (a bare list of ids) explains nothing about why those ids.
      `from_subset` fills in a default note naming the source subset when
      none is given.
    """
    given = [rule is not None for rule in (values, query, occurrence_ids, from_subset)]
    if sum(given) != 1:
        raise ValueError(
            "define_subset needs exactly one of values=, query=, "
            "occurrence_ids=, or from_subset="
        )
    if values is not None and column is None:
        raise ValueError("values= needs column= to say which column to match on")

    if from_subset is not None:
        occurrence_ids = select_ids(project_path, subset=from_subset)
        if note is None:
            note = f"frozen from subset {from_subset!r}"

    if values is not None:
        definition = {"column": column, "values": list(values)}
    elif query is not None:
        definition = {"query": query}
    else:
        # A frozen list's own digest is cheap (no table read) and never goes
        # stale the way it would for a live column/query rule, whose
        # resolution is meant to drift as the project does -- see
        # select_occurrences. Flattened rather than nested (resolved_count,
        # resolved_ids_hash): _dump_toml only writes one shallow table per
        # subset, the same {count, ids_hash} shape ids_record() gives every
        # other record in the package that names a set of occurrences.
        occurrence_ids = [str(i) for i in occurrence_ids]
        resolved = ids_record(occurrence_ids)
        definition = {"occurrence_ids": occurrence_ids,
                     "resolved_count": resolved["count"],
                     "resolved_ids_hash": resolved["ids_hash"]}

    definition["created_at"] = datetime.now(timezone.utc).isoformat()
    if note is not None:
        definition["note"] = note

    subsets = load_subsets(project_path)
    subsets[name] = definition
    _save_subsets(project_path, subsets)
    return definition


def define_subsets(project_path, column, mapping, note=None):
    """
    Define several subsets at once from one column -- the usual shape, where a
    single metadata column already separates the groups and only the names need
    tidying.

    - `column` -- occurrence column the groups are read from.
    - `mapping` -- `{column value: subset name}`, e.g.
      `{"Alabama Museum": "alabama", "AMNH": "amnh"}`. Several values may map
      to the same subset name, which merges them.
    - `note` -- optional free text, applied to every subset this call defines.
      See define_subset().
    """
    grouped = {}
    for value, name in mapping.items():
        grouped.setdefault(name, []).append(value)

    created_at = datetime.now(timezone.utc).isoformat()
    subsets = load_subsets(project_path)
    for name, values in grouped.items():
        definition = {"column": column, "values": values, "created_at": created_at}
        if note is not None:
            definition["note"] = note
        subsets[name] = definition
    _save_subsets(project_path, subsets)

    logger.info("defined %d subset(s) from column '%s': %s",
                len(grouped), column, ", ".join(sorted(grouped)))
    return {name: subsets[name] for name in grouped}


def select_occurrences(project_path, subset=None, limit=None, columns=None):
    """
    The occurrence rows a run should process.

    - `subset` -- name of a subset to narrow to, or None for the whole project.
      Every run funnels through here, so both are one code path.
    - `limit` -- optional cap applied after selection, for trying a recipe out.
    - `columns` -- occurrence columns to read; occurrence_id and any column the
      subset rule needs are added automatically.
    """
    definition = None
    if subset is not None:
        subsets = load_subsets(project_path)
        if subset not in subsets:
            raise KeyError(
                f"no subset named '{subset}' in "
                f"{paths.subsets_path(project_path)} "
                f"(defined: {sorted(subsets)})"
            )
        definition = subsets[subset]

    # A narrowed read must still carry whatever the rule needs to select on.
    # For a column/values rule that is one named column; for a query it could be
    # any of them -- the expression is arbitrary pandas -- so the only safe
    # answer is to read the whole table. Without this, select_ids() (which asks
    # for the id column alone, and which every run funnels through) raised
    # UndefinedVariableError on any query-defined subset.
    if columns is not None and definition is not None:
        if "query" in definition:
            columns = None
        elif "column" in definition:
            # Checked before the read: adding a column the table doesn't have
            # to a narrowed read fails inside pyarrow, naming the field but not
            # the subset that wanted it -- so the message below never fired.
            occurrence_records.require_columns(
                project_path, definition["column"],
                f"nothing for subset '{subset}' to select on")
            columns = list(columns) + [definition["column"]]

    df = load_occurrences(project_path, columns=columns)

    if definition is None:
        selected = df
    elif "occurrence_ids" in definition:
        wanted = {str(i) for i in definition["occurrence_ids"]}
        selected = df[df[ID_COL].isin(wanted)]
    elif "query" in definition:
        selected = df.query(definition["query"])
    else:
        column = definition["column"]
        if column not in df.columns:
            raise KeyError(
                f"subset '{subset}' selects on column '{column}', which the "
                f"occurrence table doesn't have (columns: {sorted(df.columns)})"
            )
        selected = df[df[column].isin(definition["values"])]

    if subset is not None:
        logger.info("subset '%s' selects %d of %d occurrences",
                    subset, len(selected), len(df))

    if limit is not None:
        selected = selected.head(limit)

    return selected.reset_index(drop=True)


def select_ids(project_path, subset=None, limit=None):
    """The occurrence ids select_occurrences() would return, as a list of strings."""
    return select_occurrences(project_path, subset=subset, limit=limit,
                              columns=[ID_COL])[ID_COL].tolist()


def grow_subset(project_path, name, target_size, candidate_ids=None,
                from_subset=None, seed=selectionhelpers.SAMPLE_SEED):
    """
    Define a subset if it doesn't exist yet, or grow (or shrink) it toward
    target_size otherwise, keeping every id it already holds.

    For a review or QC sample built up over several runs -- raising
    target_size later adds only the shortfall rather than resampling
    everything, the same additive guarantee ingest_occurrences(max_per_group=)
    gives a capped group on a later reimport. See selectionhelpers.grow_sample
    for the sampling rule this applies.

    - `name` -- subset to grow; created on the first call.
    - `target_size` -- desired size. Lowering it trims deterministically rather
      than raising or reshuffling who is in.
    - `candidate_ids` -- pool to draw new ids from; the whole project
      (select_ids(project_path)) if neither this nor from_subset is given. For
      a pool computed some other way, e.g. occurrences_matching(); for a named
      subset, from_subset reads more plainly and, unlike this, names the source
      in the note (below) instead of a bare count.
    - `from_subset` -- name of a subset to draw candidates from -- re-resolved
      to its CURRENT membership on every call, so growing from a live
      column/query subset picks up whatever it now matches. For capping an
      expensive pass (hand-drawn reference masks, say) to a deliberately
      smaller, independently-sized subset of a cheaper one (a screened "usable"
      set), so raising the cheap pass's size doesn't silently raise the
      expensive one's too. Mutually exclusive with candidate_ids.
    - `seed` -- passed through to grow_sample.

    Records target_size, seed, and the candidate pool as the subset's `note`
    (see define_subset), since this is the one place that opaque occurrence_ids
    list actually has a reason behind it worth writing down.

    Returns the subset's ids after growing -- the same ids now on disk.
    """
    if candidate_ids is not None and from_subset is not None:
        raise ValueError("grow_subset takes candidate_ids= or from_subset=, not both")

    try:
        already = select_ids(project_path, subset=name)
    except KeyError:
        already = []

    if from_subset is not None:
        candidate_ids = select_ids(project_path, subset=from_subset)
        pool_description = from_subset
    elif candidate_ids is not None:
        candidate_ids = list(candidate_ids)
        pool_description = f"{len(candidate_ids)} given candidate id(s)"
    else:
        candidate_ids = select_ids(project_path)
        pool_description = "whole project"

    ids = selectionhelpers.grow_sample(candidate_ids, target_size,
                                       keep_ids=already, seed=seed)
    # grow_subset already knows exactly how these ids were chosen, so it
    # records that as the note rather than leaving an opaque occurrence_ids
    # list with nothing explaining why those specimens -- see define_subset's
    # note= for the general reasoning.
    define_subset(
        project_path, name, occurrence_ids=ids,
        note=f"grow_subset(target_size={target_size}, seed={seed}, "
             f"candidate_pool={pool_description!r})")

    logger.info("subset '%s': %d -> %d occurrence(s) (target %d)",
                name, len(already), len(ids), target_size)
    return ids
