"""
Turn human labels into the filters an export should run with.

metrics.quality ships thresholds -- BLUR_WARN_VARIANCE, ASYMMETRY_WARN_SCORE,
EDGE_WARN_FRACTION -- that are eyeballed, not derived. This module replaces them
with numbers that mean something: sweep every observed value of a QC metric as a
candidate cutoff, score each one against human labels from metrics.annotation,
and hand back the winners in the shape export_metrics(filters=...) takes.

Where to sit on the precision/recall tradeoff is still the caller's judgement --
how much good data you can afford to throw away to keep bad data out is a
question about the study, not about the data. What changed is where that
judgement is stated: as a CONSTRAINT up front (max_fpr, min_precision) rather
than as a number read off a printed table by hand. Recall is then maximized
within it, because inside a budget for the cost you named, catching more bad
data is free. The sweeps are logged in full either way, so the evidence behind
each choice is there to read.

An eyeballed default is still better than no filter at all, so a metric whose
sweep can't satisfy the constraint falls back to its metrics.quality constant
and says so, rather than dropping out and quietly exporting unfiltered.

Filtering is data, not deletion. A threshold chosen here is applied at export
(export_metrics(filters=...)), so changing your mind costs one re-export and
destroys nothing.
"""

import logging

import numpy as np
import pandas as pd

from ..recipes import DEFAULT_PART
from ..export import column_name, metrics_wide
from ..metrics.quality import WARN_THRESHOLDS

logger = logging.getLogger(__name__)

# Annotation labels that mean "this occurrence should have been filtered" --
# everything except the one label that means it's fine.
BAD_FLAGS = ("not_an_organism", "cut_off", "multiple_organisms")

# How much genuinely good data a filter may throw away, unless told otherwise.
# There has to be a default constraint of SOME kind: read with no constraint at
# all, every sweep is won by the most aggressive cutoff there is, since flagging
# everything scores perfect recall. 2% is a conservative starting point, not a
# recommendation -- state your own.
DEFAULT_MAX_FPR = 0.02

# Which comparison keeps an occurrence, given the side that means "flag it".
# A sweep flags everything ABOVE the cutoff, so the export keeps everything at
# or below it -- the two must stay exact complements or the filter would exclude
# a different set than the one that was scored.
KEEP_COMPARATOR = {"above": "<=", "below": ">="}


def sweep_thresholds(df, metric_col, flag_when, flag_col, bad_flags=BAD_FLAGS):
    """
    Score every observed value of one metric column as a candidate filter
    threshold.

    df         -- wide DataFrame, one row per occurrence (see
                  export.metrics_wide). Must hold metric_col and flag_col.
    metric_col -- column of a continuous automated score.
    flag_when  -- "below" or "above": which side of the threshold counts as
                  "flag this for exclusion". Blur variance is LOWER for worse
                  images ("below"); asymmetry and edge fraction are HIGHER for
                  worse ones ("above").
    flag_col   -- column of human labels from metrics.annotation.
    bad_flags  -- label values that count as "should have been filtered".

    For every candidate threshold, computes:
      precision       -- of the occurrences flagged, the fraction genuinely bad.
      recall          -- of the genuinely bad occurrences, the fraction flagged.
      fpr             -- of the genuinely CLEAN occurrences, the fraction
                        wrongly flagged. The cost side of the tradeoff: real
                        data thrown away, which matters as much as recall.
      recall_<flag>   -- recall broken out per bad-label category, so a metric
                        that catches every cut-off organism while missing every
                        non-organism shows up as such instead of hiding behind
                        one aggregate number.
      n_bad, n_clean,
      n_<flag>        -- the counts behind each rate, so a rate computed from
                        two examples isn't mistaken for one computed from fifty.

    Rows missing either column are dropped -- you can't score what wasn't
    measured or wasn't labelled.

    Returns one row per candidate threshold, ascending. Empty if nothing has
    both columns.
    """
    if flag_when not in ("below", "above"):
        raise ValueError('flag_when must be "below" or "above"')
    for column in (metric_col, flag_col):
        if column not in df.columns:
            raise KeyError(
                f"column '{column}' not in the metrics frame "
                f"(available: {sorted(df.columns)})"
            )

    valid = df[[metric_col, flag_col]].dropna()
    valid = valid[pd.to_numeric(valid[metric_col], errors="coerce").notna()]
    if valid.empty:
        logger.warning("no rows with both %s and %s -- nothing to sweep",
                       metric_col, flag_col)
        return pd.DataFrame()

    metric = pd.to_numeric(valid[metric_col])
    flags = valid[flag_col]
    is_bad = flags.isin(bad_flags)
    n_bad = int(is_bad.sum())
    n_clean = int((~is_bad).sum())

    rows = []
    for threshold in np.sort(metric.unique()):
        flagged = (metric < threshold) if flag_when == "below" else (metric > threshold)

        true_positives = int((flagged & is_bad).sum())
        false_positives = int((flagged & ~is_bad).sum())

        row = {
            "threshold": float(threshold),
            "n_flagged": int(flagged.sum()),
            "precision": (true_positives / (true_positives + false_positives))
            if (true_positives + false_positives) else float("nan"),
            "recall": (true_positives / n_bad) if n_bad else float("nan"),
            "fpr": (false_positives / n_clean) if n_clean else float("nan"),
            "n_bad": n_bad,
            "n_clean": n_clean,
        }
        for bad_flag in bad_flags:
            in_flag = flags == bad_flag
            n_flag = int(in_flag.sum())
            row[f"recall_{bad_flag}"] = (int((flagged & in_flag).sum()) / n_flag) \
                if n_flag else float("nan")
            row[f"n_{bad_flag}"] = n_flag
        rows.append(row)

    return pd.DataFrame(rows)


def _resolve_specs(metric_specs):
    """
    Normalize the two forms of `metric_specs` into {metric_name: flag_when}.

    A bare list is allowed for the metrics metrics.quality already knows the
    direction of, because "is a higher edge_fraction worse" is a property of the
    metric rather than a decision a caller makes -- and one written out by hand
    at every call site is one that eventually gets written backwards.
    """
    if isinstance(metric_specs, dict):
        return dict(metric_specs)

    specs = {}
    for metric_name in metric_specs:
        if metric_name not in WARN_THRESHOLDS:
            raise KeyError(
                f"'{metric_name}' isn't one of the metrics with a known "
                f"direction ({sorted(WARN_THRESHOLDS)}) -- pass "
                f"metric_specs={{'{metric_name}': 'above'}} (or 'below') to say "
                "which side means flag it"
            )
        specs[metric_name] = WARN_THRESHOLDS[metric_name][1]
    return specs


def get_validated_filters(project_path, metric_specs, predicted_run,
                          annotation_run, flag_metric="annotate_flags",
                          part=DEFAULT_PART, bad_flags=BAD_FLAGS,
                          max_fpr=DEFAULT_MAX_FPR, min_precision=None,
                          defaults=None):
    """
    Calibrate several QC metrics against human labels and return the filters an
    export should run with.

    The whole path in one call: build the joined frame, sweep each metric's
    observed values as candidate cutoffs, score them against the labels, pick
    the highest-recall cutoff that satisfies the constraint, and turn each into
    an (operator, threshold) rule pointing at the right export column. The
    result goes straight into export_metrics(filters=...) -- the two halves were
    always one decision, and a caller reassembling them by hand is a caller who
    can get the comparison backwards and silently keep exactly the occurrences
    the sweep flagged.

    project_path   -- project to read from.
    metric_specs   -- which metrics to calibrate, either

                        ["edge_fraction", "bilateral_asymmetry"]
                          for metrics metrics.quality knows the direction of, or
                        {"blur_variance": "below", "edge_fraction": "above"}
                          to say which side means "flag this one" -- needed for
                          any metric of your own.

                      Bare metric names either way; the run and part prefixes
                      are added for you.
    predicted_run  -- run name the candidate QC metrics were computed under.
    annotation_run -- run name the human labels were recorded under.
    flag_metric    -- metric name holding those labels, annotate_flags by
                     default. Pass something else only for a screening metric of
                     your own; a dict-valued one needs the key too, as
                     "<metric>__<key>", since export gives each key its own
                     column.
    part           -- part both runs measured.
    bad_flags      -- passed through to sweep_thresholds.
    max_fpr,
    min_precision  -- the constraint each threshold has to satisfy (see
                     suggest_threshold). max_fpr defaults to DEFAULT_MAX_FPR
                     because an unconstrained read of a sweep always picks the
                     most aggressive cutoff available.
    defaults       -- {metric_name: threshold} to fall back on where the
                     constraint can't be met; metrics.quality.WARN_THRESHOLDS
                     for the metrics it covers if omitted. A metric with no
                     fallback and no satisfying cutoff is left OUT of the
                     returned filters and warned about -- there is nothing
                     defensible to filter it on, and inventing a number would
                     be worse than exporting it unfiltered and saying so.

    Returns {export column: (comparator, threshold)}, e.g.
    {"qc__organism__edge_fraction": ("<=", 0.031)}. Empty if nothing could be
    calibrated, which export_metrics reads as "no filtering" -- so check the log
    before trusting an unfiltered export.
    """
    specs = _resolve_specs(metric_specs)
    df = metrics_wide(project_path, run_names=[predicted_run, annotation_run],
                      parts=[part])
    if df.empty:
        logger.warning("no stored values for runs '%s'/'%s' -- no filters",
                       predicted_run, annotation_run)
        return {}

    flag_col = column_name(annotation_run, part, flag_metric)

    filters = {}
    for metric_name, flag_when in specs.items():
        if flag_when not in KEEP_COMPARATOR:
            raise ValueError(
                f"metric_specs['{metric_name}'] must be \"below\" or \"above\", "
                f"got {flag_when!r}"
            )

        metric_col = column_name(predicted_run, part, metric_name)
        sweep = sweep_thresholds(df, metric_col, flag_when, flag_col,
                                 bad_flags=bad_flags)
        if not sweep.empty:
            logger.info("%s threshold sweep (flag when %s):\n%s",
                        metric_name, flag_when, sweep.to_string(index=False))

        choice = None if sweep.empty else suggest_threshold(
            sweep, min_precision=min_precision, max_fpr=max_fpr)

        if choice is not None:
            threshold = float(choice["threshold"])
            logger.info("%s: keep %s %g -- catches %.0f%% of flagged "
                        "occurrences, discards %.1f%% of the clean ones (n=%d "
                        "bad, %d clean)", metric_name,
                        KEEP_COMPARATOR[flag_when], threshold,
                        100 * choice["recall"], 100 * choice["fpr"],
                        choice["n_bad"], choice["n_clean"])
        else:
            fallback = (defaults or {}).get(metric_name)
            if fallback is None and metric_name in WARN_THRESHOLDS:
                fallback = WARN_THRESHOLDS[metric_name][0]
            if fallback is None:
                logger.warning("%s: no cutoff satisfies the constraint and no "
                               "default to fall back on -- exporting it "
                               "unfiltered", metric_name)
                continue

            threshold = float(fallback)
            logger.warning("%s: no cutoff satisfies the constraint -- falling "
                           "back to the uncalibrated default %g", metric_name,
                           threshold)

        filters[metric_col] = (KEEP_COMPARATOR[flag_when], threshold)

    return filters


def suggest_threshold(sweep, min_precision=None, max_fpr=None):
    """
    The highest-recall threshold in a sweep that still satisfies a constraint --
    a starting point to read off a sweep table, not a recommendation.

    Give the constraint you actually care about. "Don't throw away more than 2%
    of good data" is max_fpr=0.02; "at least 80% of what I exclude should
    genuinely be bad" is min_precision=0.8. Recall is then maximized subject to
    it, because within a fixed budget for the cost you named, catching more bad
    data is free.

    Returns the winning row as a Series, or None if no threshold satisfies the
    constraints -- which is itself the answer: this metric can't separate the
    two groups well enough for what you asked.
    """
    if sweep.empty:
        return None

    candidates = sweep
    if min_precision is not None:
        candidates = candidates[candidates["precision"] >= min_precision]
    if max_fpr is not None:
        candidates = candidates[candidates["fpr"] <= max_fpr]

    if candidates.empty:
        logger.info("no threshold satisfies min_precision=%s max_fpr=%s",
                    min_precision, max_fpr)
        return None

    return candidates.loc[candidates["recall"].idxmax()]
