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
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

SPLIT_COL = "split"


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
