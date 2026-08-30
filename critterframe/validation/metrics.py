"""
Compare predicted metric values against reference values. Persists nothing.

Catches what IoU cannot: a mask can overlap well and still give the wrong
length, and a human's clicked body axis is the reference that shows it.
"""

import logging

import numpy as np
import pandas as pd

from ..export import column_name, metrics_wide
from ..recipes import DEFAULT_PART
from ..records.occurrences import ID_COL

logger = logging.getLogger(__name__)


def _pair_metrics(predicted_run, reference_run, part, metric_names, columns):
    """
    Turn the three forms of `metric_names` into one ordered list of
    (predicted metric, reference metric) pairs, so the comparison loop below
    handles a mapping, a list, and None identically instead of branching three
    ways over the actual work.

    columns -- the wide frame's column names, used to discover what the two
               runs share when metric_names is None.
    """
    if isinstance(metric_names, dict):
        return list(metric_names.items())
    if metric_names is not None:
        return [(metric_name, metric_name) for metric_name in metric_names]

    # Discovery: whatever follows "<run>__<part>__" in a wide column is a metric
    # name, or a metric name plus a dict key. Comparing the suffixes therefore
    # pairs up per-key columns as readily as scalar ones -- a dict-valued metric
    # used to coerce to NaN as a whole and drop out here.
    def suffixes(run_name):
        prefix = column_name(run_name, part, "")
        return {column[len(prefix):] for column in columns
                if column.startswith(prefix)}

    shared = sorted(suffixes(predicted_run) & suffixes(reference_run))
    return [(metric_name, metric_name) for metric_name in shared]


def compare_metrics(project_path, predicted_run, reference_run,
                    metric_names=None, part=DEFAULT_PART, current_only=True,
                    show_worst=5):
    """
    Join two runs' values per occurrence and report how far apart they are.

    project_path  -- project to read from.
    predicted_run -- run name holding the automated values.
    reference_run -- run name holding the reference values.
    metric_names  -- None compares every metric the two runs share; a list
                     compares those metrics under the same name on both sides; a
                     dict pairs {predicted_metric: reference_metric}, which is
                     what makes a human annotation comparable, e.g.
                     {"body_length": "click_two_points__length_px"}.
    part          -- part to compare.
    current_only  -- ignore values measured from masks since replaced. On by
                     default: agreement between a current reference and a stale
                     prediction reports the old segmenter's error under the new
                     one's name.
    show_worst    -- how many of the largest percent disagreements to log by id,
                     signed. 0 disables. A prediction hugely LARGER than its
                     reference is the signature of a bad reference value.

    Returns one row per pair with:
      metric, reference_metric -- what was compared against what.
      n               -- occurrences with both values present.
      mean_abs_diff   -- mean absolute difference, in the metric's own unit.
      mean_pct_diff   -- mean absolute percent difference, relative to the
                         reference. Occurrences whose reference is 0 are excluded
                         from this but still counted elsewhere.
      median_pct_diff -- the same, median instead of mean. Read together they
                         separate two questions: the median says whether the
                         population agrees, and a mean far above it says the
                         reference set contains a value that needs redoing.
      bias            -- mean signed difference, predicted minus reference. A
                         metric 5% high on every specimen is correctable; one
                         randomly 5% off in both directions is not.
      correlation     -- Pearson r. High correlation with a large bias means the
                         measurement is fine and the scale is off.
    """
    # metrics_wide rather than a pivot of its own: newest-wins, currency, and
    # the splitting of dict values into one column per key are all decisions
    # about presenting stored values, they already live there, and a second
    # implementation of them here would be free to disagree with the export
    # about what a project currently says.
    wide = metrics_wide(project_path, run_names=[predicted_run, reference_run],
                        parts=[part], current_only=current_only)
    if wide.empty:
        logger.warning("no values to compare for runs '%s'/'%s' on part '%s'",
                       predicted_run, reference_run, part)
        return pd.DataFrame()

    wide = wide.set_index(ID_COL)
    pairs = _pair_metrics(predicted_run, reference_run, part, metric_names,
                          wide.columns)
    if not pairs:
        logger.warning("runs '%s' and '%s' share no metric names",
                       predicted_run, reference_run)
        return pd.DataFrame()

    logger.info("agreement on part '%s': %s vs %s", part, predicted_run,
                reference_run)

    rows = []
    for predicted_metric, reference_metric in pairs:
        predicted_column = column_name(predicted_run, part, predicted_metric)
        reference_column = column_name(reference_run, part, reference_metric)
        label = predicted_metric if predicted_metric == reference_metric \
            else f"{predicted_metric} vs {reference_metric}"

        # A column compared against itself measures nothing, and selecting it
        # twice gives a duplicated-column frame in which every statistic below
        # degenerates into an unreadable pandas TypeError. Reached by comparing
        # a run with itself, which is a plausible typo.
        if predicted_column == reference_column:
            logger.info("  %s: predicted and reference are the same column, "
                        "skipping", label)
            continue

        missing = [column for column in (predicted_column, reference_column)
                   if column not in wide.columns]
        if missing:
            logger.info("  %s: %s not measured, skipping", label,
                        ", ".join(missing))
            continue

        pair = wide[[predicted_column, reference_column]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            logger.info("  %s: no occurrences with both values (or not numeric)",
                        label)
            continue

        difference = pair[predicted_column] - pair[reference_column]
        # Kept signed and only made absolute for the statistics: the worst
        # offenders are logged with their sign, which is what distinguishes a
        # prediction that ran long from a reference value that was mis-recorded.
        reference_values = pair[reference_column].replace(0, np.nan)
        signed_percent = (difference / reference_values) * 100
        percent = signed_percent.abs()

        excluded = len(pair) - int(percent.count())
        if excluded:
            logger.warning("  %s: %d occurrence(s) have a reference of 0; "
                           "counted in n and bias, excluded from the percent "
                           "figures", label, excluded)

        row = {
            "metric": predicted_metric,
            "reference_metric": reference_metric,
            "n": len(pair),
            "mean_abs_diff": float(difference.abs().mean()),
            "mean_pct_diff": float(percent.mean()),
            "median_pct_diff": float(percent.median()),
            "bias": float(difference.mean()),
            "correlation": float(pair[predicted_column].corr(pair[reference_column]))
            if len(pair) > 1 else float("nan"),
        }
        rows.append(row)

        logger.info("  %s: n=%d  mean|diff|=%.2f  mean|%%diff|=%.1f%%  "
                    "median|%%diff|=%.1f%%  bias=%+.2f  r=%.3f", label, row["n"],
                    row["mean_abs_diff"], row["mean_pct_diff"],
                    row["median_pct_diff"], row["bias"], row["correlation"])

        if show_worst:
            # dropna first, so an undefined ratio can't occupy an offender slot.
            worst = percent.dropna().sort_values(ascending=False).head(show_worst)
            if len(worst):
                logger.info("    worst %d: %s", len(worst),
                            ", ".join(f"{occurrence_id}={signed_percent[occurrence_id]:+.0f}%"
                                      for occurrence_id in worst.index))

    return pd.DataFrame(rows)
