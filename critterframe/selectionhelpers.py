"""
Choosing which occurrences something applies to.

Selection questions recur all over the package and are the same question every
time: out of these occurrences, which ones? A visualization wants a
representative handful, a trial run wants a few to test a recipe on, a training
split wants a held-out set, an annotation pass wants the ones a model is least
sure about. They differ in the RULE, not in the shape, so they belong together
rather than one per module.

Distinct from project.subsets, which is about NAMED, persisted selections that a
project carries in its definitions -- "the 2019 Peru material", "the specimens
Ana checked". These are transient: computed when asked for, never stored.

Everything here takes data and answers from it -- nothing reads a project, and
nothing else in the package is imported. That is deliberate rather than
incidental: this module is imported by ingest, by segmentation runs, and by
pipeline visualization, none of which should acquire a dependency on the
metrics or export layers just to sample or shard a list of ids.
export.occurrences_matching is the selection that DOES read stored values, and
it lives with the wide-form view it reads rather than here.
"""

import logging
import random

import pandas as pd

logger = logging.getLogger(__name__)

# Fixed so a project's sample is stable across runs, recipes, and sessions.
SAMPLE_SEED = 20250101


def rows_matching(df, rules):
    """
    Which rows a {column: values} rule set picks out, as a boolean Series.

    rules -- {column: value} or {column: [values...]}. A row matches when ANY
             rule matches: these name several kinds of a thing ("debris, or
             not-Lepidoptera"), not a conjunction one row must satisfy at once.

    Membership only, deliberately -- no <=, no >, no predicate. A threshold is a
    judgement about degree that a caller will want to revise, and revisable
    judgements belong where they keep the data (export filters, subsets). Keeping
    the vocabulary too small to express one is what stops it being smuggled into
    a place that can't undo it.

    A missing value never matches: not knowing what a row is cannot be the same
    as knowing it is one of these things. A named column that isn't in df raises,
    since a typo matching nothing would read as "there was none of that here".
    """
    matched = pd.Series(False, index=df.index)

    for column, values in rules.items():
        if column not in df.columns:
            raise KeyError(
                f"rule column '{column}' isn't in the table "
                f"(columns: {sorted(df.columns)})"
            )
        # A bare string is one value, not an iterable of characters -- the
        # single-value form is what anyone writes first.
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
            values = [values]

        series = df[column]
        matched |= series.isin(list(values)) & series.notna()

    return matched


def sample_occurrences(occurrence_ids, count, seed=SAMPLE_SEED):
    """
    A stable pseudo-random sample of occurrence ids, in sorted order.

    Sorted input before sampling and sorted output after, from a fixed seed: the
    same ids and the same count give the same specimens every time, in the same
    order. That is the whole point rather than an implementation detail --
    a sample that reshuffled on every call would make two visualizations of two
    versions of a recipe incomparable, and you would be unable to tell a changed
    method from a changed specimen.

    Returns every id when count exceeds how many there are, rather than raising:
    asking for 25 from a project of 8 is a reasonable thing to do.
    """
    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    if count is None or count >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, count))


def shard_occurrences(occurrence_ids, index, total):
    """
    This shard's disjoint slice of occurrence_ids, out of `total` shards.

    Sorted first, then handed out round-robin -- deterministic regardless of
    the order occurrence_ids arrives in or which process computes it, so any
    number of workers given the same ids and the same total always agree on
    the same non-overlapping split with no coordination between them. That's
    what lets a cluster job array (or a plain multiprocessing.Pool, or a
    handful of manual terminal invocations) run one shard per worker safely:
    worker i calls run_segments(..., shard=(i, n)) and never touches an
    occurrence any other worker is also touching.

    Sizes differ by at most one shard-to-shard (round-robin, not chunked), so
    no worker sits idle waiting on a shard several times the size of its
    neighbours'.

    index -- this shard's number, 0 <= index < total.
    total -- how many shards occurrence_ids is being split into.
    """
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")
    if not (0 <= index < total):
        raise ValueError(f"index must be in [0, {total}), got {index}")

    ids = sorted(str(occurrence_id) for occurrence_id in occurrence_ids)
    return ids[index::total]
