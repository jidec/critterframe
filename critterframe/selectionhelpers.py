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

Most of these take data and answer from it. occurrences_matching is the
exception -- it reads a project, because the rule it applies is written against
values a run already stored -- but the answer it gives is still transient.
"""

import logging
import random

import pandas as pd

from .export import column_name, metrics_wide
from .recipes import DEFAULT_PART
from .records.occurrences import ID_COL

logger = logging.getLogger(__name__)

# Fixed so a project's sample is stable across runs, recipes, and sessions.
SAMPLE_SEED = 20250101


def rows_matching(df, rules):
    """
    Which rows a {column: values} rule set picks out, as a boolean Series.

    rules -- {column: value} or {column: [values...]}. A row matches when ANY
             rule matches it, which is the reading these rule sets always want:
             they name several kinds of a thing ("debris, or not-Lepidoptera"),
             not a conjunction of properties one row would have to satisfy at
             once.

    Membership only, deliberately. There is no <=, no >, no predicate. A
    threshold is a judgement about degree that a caller might want to revise,
    and every mechanism in this package for revisable judgements keeps the data
    and narrows at the end (export filters, subsets). Rule sets here answer
    categorical questions -- "did the source label this row as one of these
    things" -- and keeping the vocabulary too small to express a threshold is
    what stops one being smuggled into a place that can't undo it.

    A missing value NEVER matches. Not knowing what a row is cannot be the same
    as knowing it is one of these things, and the difference matters most for
    exactly the callers that act on the answer.

    A named column that isn't in df raises, listing the ones that are: a typo
    that silently matched nothing would read as "there was none of that here",
    which is the wrong answer to have believed.
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


def occurrences_matching(project_path, run_name, rules, part=DEFAULT_PART,
                         current_only=False):
    """
    The occurrences whose stored metric values match a {metric: values} rule set
    -- "the ones a person called usable", "the ones a classifier called debris".

    project_path -- project to read from.
    run_name     -- run whose values the rules are written against.
    rules        -- {metric_name: value} or {metric_name: [values...]}, with BARE
                    metric names: the run and part prefixes are added for you
                    (export.column_name), the same convenience
                    validation.filters.get_validated_filters gives. So a
                    screening pass's usable crops are {"annotate_flags":
                    "usable"}. A dict-valued metric's keys are separate columns
                    and are named metric__key --
                    {"click_two_points__length_px": ...}.
    part         -- part the values were recorded for.
    current_only -- see below.

    Matching is rows_matching's, so its rules hold here too: any rule matches, a
    missing value never does, and a metric name that produced no column raises
    rather than selecting nothing. That last one matters most here, because the
    usual thing to do with the answer is define a subset from it, and a subset
    that is silently empty looks exactly like a review pass nobody has done yet.

    A run with no stored values at all is the one empty case that ISN'T a typo:
    it means nobody has done that pass yet, which is the normal state of a fresh
    project and is answered with an empty list and a warning. There's nothing to
    check a rule against, so the typo guard can't apply -- it applies from the
    first recorded value onward, which is when a wrong name starts being able to
    hide behind real data.

    current_only is False by default, against the grain of everything else that
    reads stored values. A selection asks what somebody SAID about these
    occurrences, and a label like "not_an_organism" or "cut_off" describes the
    CROP -- it stays true no matter which mask happened to be on screen when it
    was recorded. Left at the usual True, resegmenting would drop every label
    out of the wide view and hand back an empty selection, voiding a review
    session over a change the labels never depended on. Pass True where a rule
    reads a value that genuinely describes a mask (a mask-quality grade, an
    automated score), since there the stale value really is the wrong answer.

    Returns a sorted list of occurrence ids.
    """
    rules = {column_name(run_name, part, metric_name): values
             for metric_name, values in rules.items()}

    df = metrics_wide(project_path, run_names=[run_name], parts=[part],
                      current_only=current_only)
    if df.empty:
        logger.warning("run '%s' has no stored values for part '%s' -- nothing "
                       "to match %s against, selecting none", run_name, part,
                       sorted(rules))
        return []

    matched = df[rows_matching(df, rules)]
    logger.info("%d of %d occurrence(s) in run '%s' match %s",
                len(matched), len(df), run_name, rules)
    return sorted(matched[ID_COL].astype(str))


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
