"""
Subsets: named selections of occurrences intended to receive a particular
processing recipe.

A project is one analytical dataset -- occurrences sharing a common biological
and trait model -- but that doesn't mean every occurrence should be PROCESSED
the same way. Specimens photographed by three museums to three different
standards belong in one project (you want to compare them) while needing three
different crop regions to get at the same wings. Subsets are how that's
expressed: define the groups once, then point a different recipe at each.

Definitions live in project_path/definitions/subsets.toml, hand-editable by
design -- deciding which collection is which is human knowledge about the data,
not something derived from it, so it belongs in a small file a person can read
and correct rather than buried in a script.

A subset selects rows; it never copies or moves them. An occurrence can belong
to several subsets, or to none, and running a recipe over one subset leaves
every other subset's masks and metrics untouched.
"""

import logging

from ..records.occurrences import ID_COL, load_occurrences
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
    Every subset definition in the project, as {name: definition}. Empty if the
    project has no subsets.toml -- a project without subsets is the normal case
    and not an error.
    """
    subsets_path = paths.subsets_path(project_path)
    if not subsets_path.exists():
        return {}

    with subsets_path.open("rb") as handle:
        return tomllib.load(handle).get("subsets", {})


def save_subsets(project_path, subsets):
    """Write the whole subset table, replacing whatever was there."""
    subsets_path = paths.subsets_path(project_path)
    subsets_path.parent.mkdir(parents=True, exist_ok=True)
    subsets_path.write_text(_dump_toml(subsets))

    logger.info("wrote %d subset definition(s) -> %s", len(subsets), subsets_path)
    return subsets


def define_subset(project_path, name, column=None, values=None, query=None,
                  occurrence_ids=None):
    """
    Define (or redefine) one subset. Exactly one selection rule must be given.

    name           -- what to call it, e.g. "amnh".
    column, values -- select occurrences whose `column` is one of `values`.
                      The common case: a collection, a device, a source.
    query          -- a pandas query expression evaluated against the
                      occurrence table, e.g. "year >= 2020 and country ==
                      'Panama'". For selections a column/values pair can't
                      express.
    occurrence_ids -- an explicit list of ids. For a hand-picked selection with
                      no rule behind it -- a validation set chosen by eye, say
                      -- where writing the ids down IS the definition.
    """
    given = [rule is not None for rule in (values, query, occurrence_ids)]
    if sum(given) != 1:
        raise ValueError(
            "define_subset needs exactly one of values=, query=, or "
            "occurrence_ids="
        )
    if values is not None and column is None:
        raise ValueError("values= needs column= to say which column to match on")

    if values is not None:
        definition = {"column": column, "values": list(values)}
    elif query is not None:
        definition = {"query": query}
    else:
        definition = {"occurrence_ids": [str(i) for i in occurrence_ids]}

    subsets = load_subsets(project_path)
    subsets[name] = definition
    save_subsets(project_path, subsets)
    return definition


def define_subsets(project_path, column, mapping):
    """
    Define several subsets at once from one column -- the usual shape, where a
    single metadata column already separates the groups and only the names need
    tidying.

    column  -- occurrence column the groups are read from.
    mapping -- {column value: subset name}, e.g.
               {"Alabama Museum": "alabama", "AMNH": "amnh"}. Several values
               may map to the same subset name, which merges them.
    """
    grouped = {}
    for value, name in mapping.items():
        grouped.setdefault(name, []).append(value)

    subsets = load_subsets(project_path)
    for name, values in grouped.items():
        subsets[name] = {"column": column, "values": values}
    save_subsets(project_path, subsets)

    logger.info("defined %d subset(s) from column '%s': %s",
                len(grouped), column, ", ".join(sorted(grouped)))
    return {name: subsets[name] for name in grouped}


def select_occurrences(project_path, subset=None, limit=None, columns=None):
    """
    The occurrence rows a run should process.

    subset  -- name of a subset to narrow to, or None for the whole project.
               Every run funnels through here, so "no subset" and "this subset"
               are the same code path with the same guarantees rather than two.
    limit   -- optional cap, applied after selection. For trying a recipe on a
               handful of occurrences before committing to the whole project.
    columns -- optional list of occurrence columns to read; occurrence_id and
               any column the subset rule needs are added automatically.
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
