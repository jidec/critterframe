"""Named, persisted selections of occurrences, kept in `definitions/subsets.toml`."""

import logging
from datetime import datetime, timezone

from .. import selectionhelpers
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, ids_record, load_occurrences
from ..storage.jsonfiles import atomic_write
from . import paths

logger = logging.getLogger(__name__)

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10, via the tomli backport
    import tomli as tomllib


def _toml_value(value):
    """Serialize one scalar or list as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dump_toml(subsets):
    """Serialize the subset table as TOML."""
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
    """Return every subset definition as `{name: definition}`; empty if there is no `subsets.toml`."""
    subsets_path = paths.subsets_path(project_path)
    if not subsets_path.exists():
        return {}

    with subsets_path.open("rb") as handle:
        return tomllib.load(handle).get("subsets", {})


def _save_subsets(project_path, subsets):
    """Write the whole subset table, atomically and as UTF-8."""
    subsets_path = paths.subsets_path(project_path)
    with atomic_write(subsets_path) as handle:
        handle.write(_dump_toml(subsets))

    logger.info("wrote %d subset definition(s) -> %s", len(subsets), subsets_path)
    return subsets


def define_subset(
    project_path, name, column=None, values=None, query=None, occurrence_ids=None, from_subset=None, note=None
):
    """Define or redefine one subset, by exactly one selection rule.

    Args:
        project_path: Project to define it in.
        name: The subset's name, e.g. `"amnh"`.
        column: Occurrence column to select on, with `values`.
        values: Values of `column` that belong to the subset.
        query: A pandas query against the occurrence table, e.g. `"year >= 2020"`.
        occurrence_ids: An explicit list of ids.
        from_subset: Another subset whose current membership to freeze under this name.
        note: Free text on why the subset exists; never read back by the package.
    """
    given = [rule is not None for rule in (values, query, occurrence_ids, from_subset)]
    if sum(given) != 1:
        raise ValueError(
            "define_subset needs exactly one of values=, query=, occurrence_ids=, or from_subset="
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
        definition = {
            "occurrence_ids": occurrence_ids,
            "resolved_count": resolved["count"],
            "resolved_ids_hash": resolved["ids_hash"],
        }

    definition["created_at"] = datetime.now(timezone.utc).isoformat()
    if note is not None:
        definition["note"] = note

    subsets = load_subsets(project_path)
    subsets[name] = definition
    _save_subsets(project_path, subsets)
    return definition


def define_subsets(project_path, column, mapping, note=None):
    """Define several subsets from the values of one column.

    Args:
        project_path: Project to define them in.
        column: Occurrence column the groups are read from.
        mapping: `{column value: subset name}`. Values mapped to one name are merged.
        note: Free text applied to every subset defined.
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

    logger.info("defined %d subset(s) from column '%s': %s", len(grouped), column, ", ".join(sorted(grouped)))
    return {name: subsets[name] for name in grouped}


def select_occurrences(project_path, subset=None, limit=None, columns=None):
    """Return the occurrence rows a run should process.

    Args:
        project_path: Project to read from.
        subset: Named subset to narrow to; None for the whole project.
        limit: Cap applied after selection.
        columns: Occurrence columns to read. The id and any column the subset rule needs
            are added.
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
                project_path, definition["column"], f"nothing for subset '{subset}' to select on"
            )
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
        logger.info("subset '%s' selects %d of %d occurrences", subset, len(selected), len(df))

    if limit is not None:
        selected = selected.head(limit)

    return selected.reset_index(drop=True)


def select_ids(project_path, subset=None, limit=None):
    """Return the ids `select_occurrences` would, as a list of strings."""
    return select_occurrences(project_path, subset=subset, limit=limit, columns=[ID_COL])[ID_COL].tolist()


def grow_subset(
    project_path,
    name,
    target_size,
    candidate_ids=None,
    from_subset=None,
    seed=selectionhelpers.SAMPLE_SEED,
    candidate_note=None,
):
    """Define a subset, or grow or shrink it toward a target size, keeping the ids it holds.

    An id already in the subset but no longer in the candidate pool is dropped. The target,
    seed and pool are recorded as the subset's note.

    Args:
        project_path: Project the subset belongs to.
        name: Subset to grow; created on the first call.
        target_size: Size wanted. Raising it adds only the shortfall; lowering it trims.
        candidate_ids: Pool to draw new ids from; the whole project if neither this nor
            `from_subset` is given.
        from_subset: Subset to draw candidates from, resolved to its current membership on
            every call. Not with `candidate_ids`.
        seed: Sampling seed.
        candidate_note: What `candidate_ids` is, e.g. `"abdomens from body_parts"`, for the note.

    Returns:
        The subset's ids after growing.
    """
    if candidate_ids is not None and from_subset is not None:
        raise ValueError("grow_subset takes candidate_ids= or from_subset=, not both")
    if candidate_note is not None and candidate_ids is None:
        raise ValueError("candidate_note= describes candidate_ids=, which wasn't given")

    try:
        already = select_ids(project_path, subset=name)
    except KeyError:
        already = []

    if from_subset is not None:
        candidate_ids = select_ids(project_path, subset=from_subset)
        pool_description = from_subset
    elif candidate_ids is not None:
        candidate_ids = list(candidate_ids)
        pool_description = (
            f"{len(candidate_ids)} given candidate id(s)"
            if candidate_note is None
            else f"{candidate_note} ({len(candidate_ids)} id(s))"
        )
    else:
        candidate_ids = select_ids(project_path)
        pool_description = "whole project"

    ids = selectionhelpers.grow_sample(candidate_ids, target_size, keep_ids=already, seed=seed)
    # grow_subset already knows exactly how these ids were chosen, so it
    # records that as the note rather than leaving an opaque occurrence_ids
    # list with nothing explaining why those specimens -- see define_subset's
    # note= for the general reasoning.
    define_subset(
        project_path,
        name,
        occurrence_ids=ids,
        note=f"grow_subset(target_size={target_size}, seed={seed}, candidate_pool={pool_description!r})",
    )

    logger.info("subset '%s': %d -> %d occurrence(s) (target %d)", name, len(already), len(ids), target_size)
    return ids
