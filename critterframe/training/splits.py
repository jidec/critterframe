"""
split_ids(): grouped and stratified, to avoid leakage.

Grouping is what stops the same specimen straddling train and validation, which
would report a score for memorization. Stratifying keeps each split's class
balance close to the whole. Ids are sorted before assignment, so the seed
genuinely pins the split.
"""

import logging

import numpy as np
import pandas as pd

from ..project import paths, subsets as subset_selection
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, load_occurrences

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

SPLIT_COL = "split"


def split_ids(project_path, occurrence_ids=None, proportions=None,
              stratify_by=None, group_by=None, subset=None, seed=0):
    """
    Partition occurrence ids into named splits: {"train": [...], "val": [...]}.

    Ids in, ids out -- nothing is written. Keeping the decision separate from
    exporting images means one split can back a segmenter's dataset, an
    encoder's dataset, and a validation pass without any of them disagreeing.
    Freeze one with define_subset(..., occurrence_ids=splits["train"]).

    project_path   -- project the ids belong to.
    occurrence_ids -- the ids to split. None takes every occurrence in the
                      project, or in `subset`.
    proportions    -- {split name: proportion}, defaulting to 70/15/15
                      train/val/test. Normalized, so they needn't sum to 1. Every
                      requested name is a key of the result even when empty.
    stratify_by    -- occurrence column whose distribution to preserve across
                      splits, typically the label being trained on. A rare
                      species is the case that needs it: a random split of a
                      long-tailed table can leave it out of validation entirely.
    group_by       -- occurrence column whose members must all land on the same
                      side. The leakage guard: several images of one specimen or
                      one trap night are not independent, and splitting them
                      apart makes a validation score measure memorization.
    subset         -- restrict to a named subset instead of passing ids.
    seed           -- split seed. The same inputs give the same split whatever
                      ORDER the ids arrive in, since the frame is sorted first.

    stratify_by/group_by are occurrence columns. A label living in the metric log
    isn't reachable here -- export it onto a manifest and use split_dataset().
    """
    paths.require_project(project_path)

    if occurrence_ids is not None and subset is not None:
        raise ValueError(
            "split_ids takes occurrence_ids= or subset=, not both -- they are "
            "two ways of saying which occurrences to split"
        )

    columns = [column for column in (stratify_by, group_by) if column]
    df = _frame_to_split(project_path, occurrence_ids, subset, columns)

    proportions = dict(proportions or DEFAULT_FRACTIONS)
    assigned = split_dataset(df, fractions=proportions, group_col=group_by,
                             stratify_col=stratify_by, seed=seed)

    splits = {
        name: assigned.loc[assigned[SPLIT_COL] == name, ID_COL].tolist()
        for name in proportions
    }
    for name, ids in splits.items():
        if not ids:
            logger.warning(
                "split '%s' came out empty -- %d occurrence(s) can't be divided "
                "%d ways in the proportions asked for",
                name, len(df), len(proportions))
    return splits


def _frame_to_split(project_path, occurrence_ids, subset, columns):
    """
    The occurrence rows split_ids() will assign, carrying only the columns the
    split needs, sorted by id.

    Sorted because the assignment shuffles the groups it finds IN THE ORDER
    they appear: without this, handing the same ids to split_ids() in a
    different order (a subset query one day, a hand-written list the next)
    would silently produce a different split under the same seed, and the
    guarantee that a seed pins a split is the whole reason the seed exists.
    """
    _require_columns(project_path, columns)

    if occurrence_ids is None:
        df = subset_selection.select_occurrences(project_path, subset=subset,
                                                 columns=columns or [ID_COL])
    else:
        wanted = [str(occurrence_id) for occurrence_id in occurrence_ids]
        df = load_occurrences(project_path, columns=columns or [ID_COL])
        df = df[df[ID_COL].isin(set(wanted))]

        # An id that isn't in the table is a typo or a stale list, and silently
        # dropping it would shrink a training set without saying so.
        missing = sorted(set(wanted) - set(df[ID_COL]))
        if missing:
            raise KeyError(
                f"{len(missing)} occurrence id(s) to split aren't in the "
                f"occurrence table, e.g. {missing[:5]}"
            )

    return df.sort_values(ID_COL).reset_index(drop=True)


def _require_columns(project_path, columns):
    """
    Fail on a stratify/group column the occurrence table doesn't have, listing
    what it does have. See records.occurrences.require_columns.
    """
    occurrence_records.require_columns(project_path, columns, "nothing to split on")


def split_dataset(df, fractions=None, group_col=None, stratify_col=None,
                  seed=0, id_col="occurrence_id"):
    """
    Assign each row to a split, returning df with a `split` column added.

    df           -- manifest or occurrence table to split (e.g. from
                    training.datasets.write_dataset).
    fractions    -- {split name: fraction}, defaulting to 70/15/15
                    train/val/test. Any names and any number of splits;
                    fractions are normalized, so they don't have to sum to 1.
    group_col    -- column whose values must not be split across sides. None
                    treats every row as its own group.
    stratify_col -- column whose distribution should be preserved across
                    splits. Rows with a missing value here are pooled into one
                    stratum rather than dropped -- an unlabelled image is still
                    training data.
    seed         -- random seed, so a split is reproducible. It has to be:
                    re-splitting differently between two training runs makes
                    their validation scores incomparable, and worse, moves
                    previously-validation examples into training.
    id_col       -- column identifying a row; used only for logging.

    Returns a copy of df with the split column added.
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
        stratum_groups = [g for g in pd.unique(groups[in_stratum])
                          if g not in assignment]
        rng.shuffle(stratum_groups)
        assignment.update(_assign(stratum_groups, fractions))

    df[SPLIT_COL] = groups.map(assignment)
    _log_summary(df, fractions, group_col, stratify_col, id_col)
    return df


def _assign(group_values, fractions):
    """
    Hand out shuffled groups to splits by cumulative fraction.

    Cumulative boundaries rather than per-split counts, so rounding error can't
    leave the last split empty or the last few groups unassigned.
    """
    names = list(fractions)
    boundaries = np.cumsum([fractions[name] for name in names])
    n = len(group_values)

    assignment = {}
    for index, value in enumerate(group_values):
        position = (index + 0.5) / n if n else 0.0
        split = names[int(np.searchsorted(boundaries, position, side="right"))
                      if position < boundaries[-1] else len(names) - 1]
        assignment[value] = split
    return assignment


def split_frames(df, **kwargs):
    """
    split_dataset(), returning {split name: DataFrame} instead of one frame with
    a column -- for training code that wants the pieces separately.
    """
    split = split_dataset(df, **kwargs)
    return {name: frame.drop(columns=[SPLIT_COL]).reset_index(drop=True)
            for name, frame in split.groupby(SPLIT_COL)}


def _log_summary(df, fractions, group_col, stratify_col, id_col):
    """Report what the split actually produced, not what was asked for."""
    counts = df[SPLIT_COL].value_counts()
    detail = ", ".join(
        f"{name}={counts.get(name, 0)} ({counts.get(name, 0) / len(df):.0%}, "
        f"asked {fractions[name]:.0%})"
        for name in fractions
    )
    logger.info("split %d rows: %s", len(df), detail)

    if group_col:
        leaked = df.groupby(group_col)[SPLIT_COL].nunique()
        leaked = int((leaked > 1).sum())
        logger.info("  grouped by '%s': %d groups, %d split across sides",
                    group_col, df[group_col].nunique(), leaked)

    if stratify_col:
        for name in fractions:
            side = df[df[SPLIT_COL] == name]
            if len(side):
                logger.debug("  %s %s distribution: %s", name, stratify_col,
                             side[stratify_col].value_counts().to_dict())
