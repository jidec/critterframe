"""split_ids(): grouped and stratified splits of occurrence ids."""

import logging

import numpy as np
import pandas as pd

from ..project import paths
from ..selection import subsets as subset_selection
from ..core.recipes import hash_spec
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, ids_record, load_occurrences
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

SPLIT_COL = "split"


# How many of a stratify column's classes a split figure draws by name; the rest
# are pooled, so a long-tailed label still gives a readable chart.
FIGURE_CLASSES = 20


def split_ids(
    project_path,
    occurrence_ids=None,
    fractions=None,
    stratify_col=None,
    group_col=None,
    subset=None,
    seed=0,
    visualize=True,
):
    """Partition occurrence ids into named splits, e.g. `{"train": [...], "val": [...]}`.

    Nothing is written; freeze a split with `define_subset(..., occurrence_ids=splits["train"])`.

    Args:
        project_path: Project the ids belong to.
        occurrence_ids: Ids to split; None takes every occurrence in the project or `subset`.
        fractions: `{split name: fraction}`, normalized; 70/15/15 train/val/test if None.
        stratify_col: Occurrence column whose distribution each split should preserve.
        group_col: Occurrence column whose members must all land in one split.
        subset: Named subset to split.
        seed: Split seed; the result does not depend on the order ids are given in.
        visualize: Write a pipeline figure of occurrences per split.

    Returns:
        `{split name: sorted ids}`, with every requested name present even when empty.
    """
    paths.require_project(project_path)

    if occurrence_ids is not None and subset is not None:
        raise ValueError(
            "split_ids takes occurrence_ids= or subset=, not both -- they are "
            "two ways of saying which occurrences to split"
        )

    columns = [column for column in (stratify_col, group_col) if column]
    df = _frame_to_split(project_path, occurrence_ids, subset, columns)

    fractions = dict(fractions or DEFAULT_FRACTIONS)
    assigned = split_dataset(
        df, fractions=fractions, group_col=group_col, stratify_col=stratify_col, seed=seed
    )

    splits = {name: assigned.loc[assigned[SPLIT_COL] == name, ID_COL].tolist() for name in fractions}
    for name, ids in splits.items():
        if not ids:
            logger.warning(
                "split '%s' came out empty -- %d occurrence(s) can't be divided "
                "%d ways in the fractions asked for",
                name,
                len(df),
                len(fractions),
            )

    identity = {
        "kind": "split_ids",
        "fractions": fractions,
        "stratify_col": stratify_col,
        "group_col": group_col,
        "seed": seed,
        "occurrences": ids_record(df[ID_COL]),
    }
    with pipeline_visualization.open_report(
        project_path, "split_ids", hash_spec(identity), visualize=visualize, identity=identity
    ).begin([]) as report:
        if report and len(assigned):
            report.figure("counts", _split_figure(assigned, fractions, stratify_col))
    return splits


def _split_figure(assigned, fractions, stratify_col):
    """Return a chart of occurrences per split, stacked by the stratify column's commonest classes."""
    if not stratify_col:
        counts = assigned[SPLIT_COL].value_counts()
        return figures.bar_chart(
            {name: int(counts.get(name, 0)) for name in fractions},
            ylabel="occurrences",
            title="occurrences per split",
        )

    labels = assigned[stratify_col].fillna("(none)").astype(str)
    named = set(labels.value_counts().head(FIGURE_CLASSES).index)
    labels = labels.where(labels.isin(named), "(other)")
    table = pd.crosstab(assigned[SPLIT_COL], labels)
    counts = {
        name: {label: int(table.at[name, label]) for label in table.columns} if name in table.index else {}
        for name in fractions
    }
    return figures.bar_chart(
        counts, ylabel="occurrences", title=f"occurrences per split, by '{stratify_col}'"
    )


def _frame_to_split(project_path, occurrence_ids, subset, columns):
    """Return the occurrence rows to assign, with only the needed columns, sorted by id."""
    _require_columns(project_path, columns)

    if occurrence_ids is None:
        df = subset_selection.select_occurrences(project_path, subset=subset, columns=columns or [ID_COL])
    else:
        wanted = [str(occurrence_id) for occurrence_id in occurrence_ids]
        df = load_occurrences(project_path, columns=columns or [ID_COL])
        df = df[df[ID_COL].isin(set(wanted))]

        # An id that isn't in the table is a typo or a stale list, and silently
        # dropping it would shrink a training set without saying so.
        missing = sorted(set(wanted) - set(df[ID_COL]))
        if missing:
            raise KeyError(
                f"{len(missing)} occurrence id(s) to split aren't in the occurrence table, e.g. {missing[:5]}"
            )

    return df.sort_values(ID_COL).reset_index(drop=True)


def _require_columns(project_path, columns):
    """Raise for a stratify or group column the occurrence table doesn't have."""
    occurrence_records.require_columns(project_path, columns, "nothing to split on")


def split_dataset(df, fractions=None, group_col=None, stratify_col=None, seed=0, id_col="occurrence_id"):
    """Assign each row of a table to a split.

    Args:
        df: Manifest or occurrence table.
        fractions: `{split name: fraction}`, normalized; 70/15/15 train/val/test if None.
        group_col: Column whose values must not be split across sides; None treats each row
            as its own group.
        stratify_col: Column whose distribution to preserve. Rows with a missing value form
            one stratum.
        seed: Random seed.
        id_col: Column identifying a row, used for logging.

    Returns:
        A copy of `df` with a `split` column.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("fractions must sum to something positive")
    fractions = {name: value / total for name, value in fractions.items()}

    if df.empty:
        return df.assign(**{SPLIT_COL: pd.Series(dtype="object")})

    df = df.copy().reset_index(drop=True)
    groups = df[group_col] if group_col else pd.Series(df.index, index=df.index)

    if stratify_col:
        strata = df[stratify_col].fillna("__unlabelled__")
    else:
        strata = pd.Series("__all__", index=df.index)

    rng = np.random.default_rng(seed)
    assignment = {}

    for stratum in sorted(strata.unique(), key=str):
        in_stratum = strata == stratum
        # A group is assigned as a unit, so it must belong to one stratum; where
        # a group spans strata, its first stratum claims it and later ones skip
        # it -- which keeps the no-leakage guarantee exact at the cost of some
        # stratum balance, the tradeoff described in the module docstring.
        stratum_groups = [g for g in pd.unique(groups[in_stratum]) if g not in assignment]
        rng.shuffle(stratum_groups)
        assignment.update(_assign(stratum_groups, fractions))

    df[SPLIT_COL] = groups.map(assignment)
    _log_summary(df, fractions, group_col, stratify_col, id_col)
    return df


def _assign(group_values, fractions):
    """Hand out shuffled groups to splits by cumulative fraction.

    Cumulative boundaries, not per-split counts, so rounding can't leave the last split empty.
    """
    names = list(fractions)
    boundaries = np.cumsum([fractions[name] for name in names])
    n = len(group_values)

    assignment = {}
    for index, value in enumerate(group_values):
        position = (index + 0.5) / n if n else 0.0
        split = names[
            int(np.searchsorted(boundaries, position, side="right"))
            if position < boundaries[-1]
            else len(names) - 1
        ]
        assignment[value] = split
    return assignment


def split_frames(df, **kwargs):
    """Return `split_dataset`'s result as `{split name: DataFrame}`."""
    split = split_dataset(df, **kwargs)
    return {
        name: frame.drop(columns=[SPLIT_COL]).reset_index(drop=True)
        for name, frame in split.groupby(SPLIT_COL)
    }


def _log_summary(df, fractions, group_col, stratify_col, id_col):
    """Log what the split produced."""
    counts = df[SPLIT_COL].value_counts()
    detail = ", ".join(
        f"{name}={counts.get(name, 0)} ({counts.get(name, 0) / len(df):.0%}, asked {fractions[name]:.0%})"
        for name in fractions
    )
    logger.info("split %d rows: %s", len(df), detail)

    if group_col:
        leaked = df.groupby(group_col)[SPLIT_COL].nunique()
        leaked = int((leaked > 1).sum())
        logger.info(
            "  grouped by '%s': %d groups, %d split across sides", group_col, df[group_col].nunique(), leaked
        )

    if stratify_col:
        for name in fractions:
            side = df[df[SPLIT_COL] == name]
            if len(side):
                logger.debug(
                    "  %s %s distribution: %s",
                    name,
                    stratify_col,
                    side[stratify_col].value_counts().to_dict(),
                )
