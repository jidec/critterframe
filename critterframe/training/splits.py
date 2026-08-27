"""
Training/validation/test splits.

Splitting looks trivial and isn't, because biological image datasets leak in
ways a random split doesn't catch. The two failure modes worth designing
against:

  GROUPED LEAKAGE. Several images of the same individual, the same specimen
  drawer, the same trap night, or the same iNaturalist observation are not
  independent. Split them at random and near-duplicates land on both sides,
  and the validation score measures memorization rather than generalization.
  Pass group_col to keep every member of a group on one side.

  IMBALANCE. Species counts in field data are wildly uneven. A random split of
  a set where 60% of images are one common species can leave a rare one absent
  from validation entirely. Pass stratify_col to preserve each category's
  proportions across the splits.

Both at once is supported and is usually what you want (group by specimen,
stratify by species): groups are assigned whole, stratum by stratum, so the
grouping guarantee holds exactly while the stratification holds approximately
-- an inherent tension, since a group can't be divided to balance a stratum,
and the leakage guarantee is the one worth keeping exact.

Two entry points, differing only in what they take and give back. split_ids()
is the one a pipeline script calls: project in, {split name: occurrence ids}
out, deciding nothing about what is then done with them. split_dataset() takes
a frame you already have and adds a column to it, which is what the manifest-
shaped callers want. Neither writes anything -- a split is a decision about the
data, and freezing one is define_subset()'s job, not this module's.
"""

import logging

import numpy as np
import pandas as pd

from ..project import paths, subsets as subset_selection
from ..records.occurrences import ID_COL, load_occurrences
from ..storage.tables import table_columns

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

SPLIT_COL = "split"


def split_ids(project_path, occurrence_ids=None, proportions=None,
              stratify_by=None, group_by=None, subset=None, seed=0):
    """
    Partition occurrence ids into named splits: {"train": [...], "val": [...]}.

    Ids in, ids out. Nothing is written and nothing is materialized -- what a
    split IS is a decision about which occurrences answer which question, and
    keeping that separate from exporting images means the same decision can be
    re-used for a segmenter's dataset, an encoder's dataset, and a validation
    pass without any of them being able to disagree about it. Freeze one with
    define_subset(project_path, "train", occurrence_ids=splits["train"]) when it
    should outlive the script that made it.

    project_path   -- project the ids belong to. Read for the stratify/group
                      columns, and to check the ids exist.
    occurrence_ids -- the ids to split. None takes every occurrence in the
                      project (or in `subset`).
    proportions    -- {split name: proportion}, defaulting to 70/15/15
                      train/val/test. Any names and any number of splits;
                      normalized, so they needn't sum to 1. Every requested
                      name is a key of the result even when it came out empty,
                      so a caller looping over splits never has to guess which
                      ones exist.
    stratify_by    -- occurrence column whose distribution to preserve across
                      splits, typically the label being trained on. A rare
                      species is the case that needs it: a random split of a
                      long-tailed table can leave it out of validation
                      entirely.
    group_by       -- occurrence column whose members must all land on the same
                      side. The leakage guard: several images of one specimen,
                      one observation, or one trap night are not independent,
                      and splitting them apart makes a validation score measure
                      memorization. Exact where stratification is approximate,
                      since a group can't be divided (see the module docstring).
    subset         -- restrict to a named subset instead of passing ids.
                      Mutually exclusive with occurrence_ids.
    seed           -- split seed. The same ids, proportions, columns, and seed
                      give the same split whatever ORDER the ids arrive in --
                      the frame is sorted before assignment -- so a split
                      survives being recomputed by a different script.

    stratify_by/group_by are occurrence columns, which is how the rest of the
    package names a grouping (outliers.group_col, calibration scopes, subset
    column). A label that lives in the metric log rather than on the occurrence
    table isn't reachable here; pass the ids per class explicitly, or export it
    onto a manifest and use split_dataset().
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
    what it does have.

    Read from the parquet footer rather than by catching a read error, so the
    message names the columns available instead of restating the one that
    wasn't -- a mistyped stratify_by is nearly always a near-miss on a column
    that IS there.
    """
    if not columns:
        return
    available = table_columns(paths.occurrences_path(project_path))
    unknown = [column for column in columns if column not in available]
    if unknown:
        raise KeyError(
            f"occurrence table has no column(s) {unknown} to split on "
            f"(columns: {sorted(available)})"
        )


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
