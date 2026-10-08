"""
Calibrate thresholds against human labels, returning the filters an export
should run with, and audit a finished filter set against labels it never saw.

Sweep each metric's observed values as candidate cutoffs, score them against
the labels, and pick the highest-recall cutoff satisfying a constraint. The
constraint matters: an unconstrained sweep always "wins" with the most
aggressive cutoff, which excludes everything.
"""

import logging
import math

import numpy as np
import pandas as pd

from ..recipes import DEFAULT_PART, hash_spec
from ..export import (
    _apply_filters,
    _recorded_filters,
    column_name,
    export_metrics,
    metrics_wide,
)
from ..metrics.quality import WARN_THRESHOLDS
from ..project.subsets import select_ids
from ..records.occurrences import ID_COL
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization
from .filter_grids import draw_filter_grids

logger = logging.getLogger(__name__)

# Below this many labelled rows, or this many bad ones, an audit's intervals are
# too wide to support a claim, and the log says so.
MIN_AUDIT_ROWS = 50
MIN_AUDIT_BAD = 10

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

# What a candidate in `metric_specs` can be: a continuous score flagged on one
# side, or a categorical value whose worst categories are dropped.
DIRECTIONS = ("below", "above", "category")

# Fewer labels than this and a category's bad rate is too noisy to drop it on.
DEFAULT_MIN_LABELLED = 10

# The constraint values a calibration also reports the outcome of, beside the
# ones it was called with: what each setting would have kept and removed.
TRADEOFF_MAX_FPR = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, None)
TRADEOFF_MIN_PRECISION = (None, 0.5, 0.6, 0.7, 0.8, 0.9)


def sweep_thresholds(df, metric_col, flag_when, label_col, bad_labels):
    """
    Score every observed value of one metric column as a candidate filter
    threshold.

    - `df` -- wide DataFrame, one row per occurrence, holding `metric_col`
      and `label_col`.
    - `metric_col` -- column of a continuous automated score.
    - `flag_when` -- `"below"` or `"above"`: which side counts as "flag this
      for exclusion". Blur variance is lower for worse images (`"below"`);
      asymmetry and edge fraction are higher for worse ones.
    - `label_col` -- column of human labels from `metrics.annotation`.
    - `bad_labels` -- label values that count as "should have been
      filtered".

    Computes per candidate threshold:

    - `precision` -- of those flagged, the fraction genuinely bad.
    - `recall` -- of the genuinely bad, the fraction flagged.
    - `fpr` -- of the genuinely clean, the fraction wrongly flagged. The
      cost side: real data thrown away.
    - `recall_<label>` -- recall per bad-label category, so a metric that
      catches every cut-off organism while missing every non-organism shows
      up instead of hiding behind one number.
    - `n_bad`, `n_clean`, `n_<label>` -- the counts behind each rate, so a
      rate from two examples isn't mistaken for one from fifty.

    Rows missing either column are dropped. Returns one row per candidate
    threshold, ascending; empty if nothing has both columns.
    """
    if flag_when not in ("below", "above"):
        raise ValueError('flag_when must be "below" or "above"')
    for column in (metric_col, label_col):
        if column not in df.columns:
            raise KeyError(
                f"column '{column}' not in the metrics frame "
                f"(available: {sorted(df.columns)})"
            )

    valid = df[[metric_col, label_col]].dropna()
    valid = valid[pd.to_numeric(valid[metric_col], errors="coerce").notna()]
    if valid.empty:
        logger.warning("no rows with both %s and %s -- nothing to sweep",
                       metric_col, label_col)
        return pd.DataFrame()

    metric = pd.to_numeric(valid[metric_col])
    flags = valid[label_col]
    is_bad = flags.isin(bad_labels)
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
        for bad_label in bad_labels:
            in_flag = flags == bad_label
            n_flag = int(in_flag.sum())
            row[f"recall_{bad_label}"] = (int((flagged & in_flag).sum()) / n_flag) \
                if n_flag else float("nan")
            row[f"n_{bad_label}"] = n_flag
        rows.append(row)

    return pd.DataFrame(rows)


def _resolve_bad_labels(labels, bad_labels, good_labels):
    """
    The label values that count as bad, from whichever way the caller said it.

    `good_labels` names the complement: every label present that isn't one of
    them. `bad_labels` is used as given. One of the two is required: which
    labels are bad is the project's own vocabulary.
    """
    _require_one_label_set(bad_labels, good_labels)
    if good_labels is not None:
        return sorted(set(labels.dropna().map(str)) - {str(label) for label in good_labels})
    return list(bad_labels)


def _require_one_label_set(bad_labels, good_labels):
    if bad_labels is not None and good_labels is not None:
        raise ValueError("give bad_labels= or good_labels=, not both -- they are "
                         "two ways of drawing one line")
    if bad_labels is None and good_labels is None:
        raise ValueError("give bad_labels= or good_labels= -- which labels mean "
                         "\"should have been filtered\" is the project's own vocabulary")


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


def _resolve_runs(metric_specs, predicted_run):
    """`metric_specs` as {run: {metric_name: direction}}, whichever form it arrived in."""
    by_run = (isinstance(metric_specs, dict) and bool(metric_specs)
              and all(isinstance(specs, (dict, list, tuple))
                      for specs in metric_specs.values()))
    if by_run:
        if predicted_run is not None:
            raise ValueError(
                "metric_specs is keyed by run, so predicted_run has nothing to "
                "name -- leave it out")
        return {run: _resolve_specs(specs) for run, specs in metric_specs.items()}
    if predicted_run is None:
        raise ValueError(
            "get_validated_filters needs predicted_run=, or metric_specs keyed "
            "by run: {run: {metric: 'below' | 'above' | 'category'}}")
    return {predicted_run: _resolve_specs(metric_specs)}


def _plain_category(value):
    """One category as a plain Python value: a whole-number float is the integer it stands for."""
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _sweep_categories(df, metric_col, label_col, bad_labels,
                      min_labelled=DEFAULT_MIN_LABELLED):
    """
    Score dropping the worst k categories of one categorical column, for every k.

    Categories with at least `min_labelled` labels are ordered by labelled bad
    rate, worst first; candidate k drops the first k. The rates are the ones
    `sweep_thresholds` reports, so `suggest_threshold` reads either.

    - `df` -- wide DataFrame holding `metric_col` and `label_col`.
    - `metric_col` -- column of a categorical value, e.g. a cluster id.
    - `label_col` -- column of human labels.
    - `bad_labels` -- label values that count as "should have been filtered".
    - `min_labelled` -- fewer labels than this and a category is never dropped.

    Returns `(sweep, table)`: one row per k, ascending from 0, with `threshold`
    holding k and `dropped` the categories; and one row per category with `n`,
    `n_bad`, `bad_rate`, `enough_labels`. Both empty if nothing has both columns.
    """
    for column in (metric_col, label_col):
        if column not in df.columns:
            raise KeyError(
                f"column '{column}' not in the metrics frame "
                f"(available: {sorted(df.columns)})"
            )

    labelled = df[[metric_col, label_col]].dropna()
    if labelled.empty:
        logger.warning("no rows with both %s and %s -- nothing to sweep",
                       metric_col, label_col)
        return pd.DataFrame(), pd.DataFrame()

    is_bad = labelled[label_col].isin(bad_labels)
    n_bad = int(is_bad.sum())
    n_clean = int((~is_bad).sum())

    table = pd.DataFrame([
        {"category": category, "n": len(group), "n_bad": int(is_bad[group.index].sum())}
        for category, group in labelled.groupby(metric_col)])
    table["bad_rate"] = table["n_bad"] / table["n"]
    table["enough_labels"] = table["n"] >= min_labelled
    # Worst first; ties go to the category with more bad labels behind its rate,
    # then by name so the order never depends on how the rows arrived.
    table["_name"] = table["category"].map(str)
    table = table.sort_values(["bad_rate", "n_bad", "_name"],
                              ascending=[False, False, True]).drop(columns="_name")
    eligible = table.loc[table["enough_labels"], "category"].tolist()

    rows = []
    for k in range(len(eligible) + 1):
        flagged = labelled[metric_col].isin(eligible[:k])
        true_positives = int((flagged & is_bad).sum())
        false_positives = int((flagged & ~is_bad).sum())
        rows.append({
            "threshold": k,
            "dropped": [_plain_category(category) for category in eligible[:k]],
            "n_flagged": int(flagged.sum()),
            "precision": (true_positives / (true_positives + false_positives))
            if (true_positives + false_positives) else float("nan"),
            "recall": (true_positives / n_bad) if n_bad else float("nan"),
            "fpr": (false_positives / n_clean) if n_clean else float("nan"),
            "n_bad": n_bad,
            "n_clean": n_clean,
        })

    table["category"] = table["category"].map(_plain_category)
    return pd.DataFrame(rows), table.reset_index(drop=True)


def _pick(sweep, max_fpr, min_precision):
    """
    The highest-recall row of a sweep within a constraint, and why there is none.

    Returns `(row, reason)`: reason is None with a row, `"none satisfy"` where
    no candidate meets the constraint, `"no bad labels"` where recall is
    undefined everywhere. Silent, so a grid of constraints can call it freely.
    """
    if sweep.empty:
        return None, "none satisfy"

    candidates = sweep
    if min_precision is not None:
        candidates = candidates[candidates["precision"] >= min_precision]
    if max_fpr is not None:
        candidates = candidates[candidates["fpr"] <= max_fpr]
    if candidates.empty:
        return None, "none satisfy"
    if candidates["recall"].isna().all():
        return None, "no bad labels"
    return candidates.loc[candidates["recall"].idxmax()], None


def _choose_rule(sweep, direction, metric_name, max_fpr, min_precision, defaults):
    """
    The rule one candidate gets under one constraint, from a sweep already made.

    - `sweep` -- from `sweep_thresholds`, or `_sweep_categories` for a
      `"category"` direction.
    - `direction` -- `"below"`, `"above"` or `"category"`.
    - `metric_name` -- bare metric name, for the `defaults` / `WARN_THRESHOLDS` fallback.
    - `max_fpr`, `min_precision` -- the constraint.
    - `defaults` -- `{metric_name: threshold}` fallbacks.

    Returns `(rule, why, choice)`. `why` is `"chosen"`, `"fallback"` (an
    uncalibrated default), `"zero recall"` (the best rule within the constraint
    catches no bad label, so there is none) or `"no cutoff"`. `choice` is the
    winning sweep row where one was chosen. Silent: choosing for another
    constraint is a lookup, and a grid of them calls this many times.
    """
    choice, _reason = _pick(sweep, max_fpr, min_precision)

    if direction == "category":
        if choice is not None and choice["recall"] > 0:
            return ("not in", list(choice["dropped"])), "chosen", choice
        return None, ("zero recall" if choice is not None else "no cutoff"), None

    if choice is not None and choice["recall"] > 0:
        return (KEEP_COMPARATOR[direction], float(choice["threshold"])), "chosen", choice
    if choice is not None:
        return None, "zero recall", None

    fallback = (defaults or {}).get(metric_name)
    if fallback is None and metric_name in WARN_THRESHOLDS:
        fallback = WARN_THRESHOLDS[metric_name][0]
    if fallback is None:
        return None, "no cutoff", None
    return (KEEP_COMPARATOR[direction], float(fallback)), "fallback", None


def _calibrate_threshold(df, metric_col, metric_name, label, flag_when, label_col,
                         bad_labels, max_fpr, min_precision, defaults, report):
    """
    The `(comparator, threshold)` rule for one continuous metric, or None where
    the labels justify none; and the sweep it was read from.
    """
    sweep = sweep_thresholds(df, metric_col, flag_when, label_col, bad_labels=bad_labels)
    if not sweep.empty:
        logger.info("%s threshold sweep (flag when %s):\n%s",
                    label, flag_when, sweep.to_string(index=False))

    rule, why, choice = _choose_rule(sweep, flag_when, metric_name, max_fpr,
                                     min_precision, defaults)
    marks = None
    if why == "chosen":
        logger.info("%s: keep %s %g -- catches %.0f%% of flagged "
                    "occurrences, discards %.1f%% of the clean ones (n=%d "
                    "bad, %d clean)", label, rule[0], rule[1],
                    100 * choice["recall"], 100 * choice["fpr"],
                    choice["n_bad"], choice["n_clean"])
        marks = {f"chosen {rule[1]:g}": rule[1]}
    elif why == "zero recall":
        # The best cutoff within the constraint catches nothing. It would still
        # sit at the edge of the labelled range and remove unlabelled
        # occurrences beyond it, on no evidence.
        logger.warning("%s: no cutoff within the constraint catches a single "
                       "bad label -- it separates nothing here, so it is left "
                       "out", label)
    elif why == "fallback":
        logger.warning("%s: no cutoff satisfies the constraint -- falling "
                       "back to the uncalibrated default %g", label, rule[1])
        marks = {f"fallback {rule[1]:g}": rule[1]}
    else:
        logger.warning("%s: no cutoff satisfies the constraint and no "
                       "default to fall back on -- exporting it "
                       "unfiltered", label)

    if report and not sweep.empty:
        report.figure(label, figures.line_chart(
            {rate: (sweep["threshold"].tolist(), sweep[rate].tolist())
             for rate in ("recall", "fpr", "precision")},
            xlabel=f"{label} threshold (flag when {flag_when})", ylabel="rate",
            title=f"{label}: n={int(sweep['n_bad'].iloc[0])} bad, "
                  f"{int(sweep['n_clean'].iloc[0])} clean",
            marks=marks))
    return rule, sweep


def _calibrate_category(df, metric_col, label, label_col, bad_labels, max_fpr,
                        min_precision, min_labelled, report):
    """
    The `("not in", categories)` rule for one categorical metric, or None where
    the labels justify none; and the sweep it was read from.
    """
    sweep, table = _sweep_categories(df, metric_col, label_col, bad_labels=bad_labels,
                                     min_labelled=min_labelled)
    if sweep.empty:
        return None, sweep
    logger.info("%s by category, worst first (droppable with at least %d "
                "labels):\n%s", label, min_labelled, table.to_string(index=False))

    sparse = table[~table["enough_labels"] & (table["bad_rate"] > 0.5)]
    if not sparse.empty:
        logger.warning("%s: %s mostly bad on fewer than %d label(s) -- kept; "
                       "label more to decide", label, sparse["category"].tolist(),
                       min_labelled)

    rule, _why, choice = _choose_rule(sweep, "category", None, max_fpr, min_precision,
                                      None)
    dropped = [] if rule is None else rule[1]

    if dropped:
        logger.info("%s: drop %s -- catches %.0f%% of flagged occurrences, "
                    "discards %.1f%% of the clean ones (n=%d bad, %d clean)",
                    label, dropped, 100 * choice["recall"], 100 * choice["fpr"],
                    choice["n_bad"], choice["n_clean"])
    else:
        logger.warning("%s: dropping no set of categories catches a bad label "
                       "within the constraint -- left out", label)

    if report:
        report.figure(label, figures.bar_chart(
            {str(row.category): {"good": row.n - row.n_bad, "bad": row.n_bad}
             for row in table.itertuples(index=False)},
            xlabel=f"{label}, worst first", ylabel="labelled rows",
            title=f"{label}: dropped {dropped or 'none'}"))

    return rule, sweep


def _tradeoff_filters(candidates, labelled, label_col, bad_labels, defaults,
                      drop_redundant, own):
    """
    The filter set each combination of `max_fpr` and `min_precision` would give.

    - `candidates` -- `[(column, metric_name, direction, sweep)]`, the sweeps
      already made for this call.
    - `labelled`, `label_col`, `bad_labels` -- the calibration labels, for
      `drop_redundant`.
    - `own` -- `(max_fpr, min_precision)` the call was made with, added to the grid.

    Returns `[(max_fpr, min_precision, filters)]` in grid order. The combination
    with neither constraint is left out: unconstrained, every sweep is won by
    removing everything.
    """
    def values(grid, mine):
        return list(dict.fromkeys([*grid, mine]))

    combinations = []
    for max_fpr in values(TRADEOFF_MAX_FPR, own[0]):
        for min_precision in values(TRADEOFF_MIN_PRECISION, own[1]):
            if max_fpr is None and min_precision is None:
                continue
            filters = {}
            for column, metric_name, direction, sweep in candidates:
                rule, _why, _choice = _choose_rule(sweep, direction, metric_name,
                                                   max_fpr, min_precision, defaults)
                if rule is not None:
                    filters[column] = rule
            if drop_redundant and len(filters) > 1:
                filters = _drop_redundant(labelled, filters, label_col, bad_labels,
                                          quiet=True)
            combinations.append((max_fpr, min_precision, filters))
    return combinations


def _tradeoff_table(combinations, labelled, label_col, bad_labels):
    """One row per combination: what its filter set does to `labelled`."""
    rows = []
    for max_fpr, min_precision, filters in combinations:
        result = _score_filters(labelled, filters, label_col, bad_labels)[0]
        rows.append({
            "setting": _constraint_label(max_fpr, min_precision),
            "max_fpr": max_fpr,
            "min_precision": min_precision,
            "n_filters": len(filters),
            "good_retained": result["good_retained"],
            "bad_caught": result["bad_caught"],
            "bad_rate_after": result["bad_rate_after"],
            "coverage": result["coverage"],
        })
    return pd.DataFrame(rows)


def _frontier(table):
    """
    The distinct outcomes of a tradeoff table, with the ones worth choosing between marked.

    Combinations with the same `good_retained` and `bad_caught` are one
    outcome: the first in grid order, with `n_same` counting the rest. An
    outcome is on the frontier when no other keeps at least as many good rows
    AND removes at least as many bad ones, with one of the two strictly better.

    Returns the distinct rows with `n_same` and `frontier` columns added.
    """
    scored = table.dropna(subset=["good_retained", "bad_caught"])
    if scored.empty:
        return scored.assign(n_same=pd.Series(dtype=int), frontier=pd.Series(dtype=bool))

    keys = ["good_retained", "bad_caught"]
    distinct = scored.drop_duplicates(subset=keys).copy()
    counts = scored.groupby(keys).size()
    distinct["n_same"] = [int(counts[(row.good_retained, row.bad_caught)]) - 1
                          for row in distinct.itertuples()]

    good = distinct["good_retained"].to_numpy()
    bad = distinct["bad_caught"].to_numpy()
    beaten = [bool((((good >= g) & (bad >= b)) & ((good > g) | (bad > b))).any())
              for g, b in zip(good, bad)]
    distinct["frontier"] = [not lost for lost in beaten]
    return distinct.reset_index(drop=True)


def _constraint_label(max_fpr, min_precision):
    """A combination as short text, e.g. `fpr<=0.05 prec>=0.6`."""
    pieces = []
    if max_fpr is not None:
        pieces.append(f"fpr<={max_fpr:g}")
    if min_precision is not None:
        pieces.append(f"prec>={min_precision:g}")
    return " ".join(pieces)


def _tradeoff_figure(table, own, title):
    """Good rows kept against bad rows removed, one point per distinct outcome."""
    outcomes = _frontier(table)
    mine = table[table["setting"] == _constraint_label(*own)]
    mine = (None if mine.empty else
            (mine.iloc[0]["good_retained"], mine.iloc[0]["bad_caught"]))

    groups, annotations = [], []
    for row in outcomes.itertuples():
        is_mine = mine is not None and (row.good_retained, row.bad_caught) == mine
        groups.append("this call" if is_mine
                      else "best tradeoffs" if row.frontier else "beaten by another")
        # A point stands for every setting with that outcome. It is named by
        # the first in grid order, or by this call's own setting where it is one.
        label = f"{_constraint_label(*own)} (this call)" if is_mine else row.setting
        if row.n_same:
            label += f" +{row.n_same}"
        annotations.append(label if row.frontier or is_mine else None)

    return figures.scatter(
        (100 * outcomes["good_retained"]).tolist(), (100 * outcomes["bad_caught"]).tolist(),
        groups=groups, annotations=annotations,
        xlabel="good rows kept (%)", ylabel="bad rows removed (%)", title=title)


def _draw_tradeoffs(report, combinations, labelled, label_col, bad_labels, own, name, what):
    """Log the best tradeoffs among `combinations` on `labelled` and draw them all."""
    if labelled.empty:
        return
    table = _tradeoff_table(combinations, labelled, label_col, bad_labels)
    outcomes = _frontier(table)
    best = outcomes[outcomes["frontier"]].sort_values("good_retained", ascending=False)
    if best.empty:
        # With no bad label, or no good one, a rate behind the chart is
        # undefined and there is no tradeoff to show.
        logger.info("%s has no tradeoff to show %s -- it needs both good and bad "
                    "labels", name, what)
        return
    logger.info(
        "what other constraints would give, %s -- the settings no other beats on "
        "both good rows kept and bad rows removed (n_same: other settings with "
        "the same outcome):\n%s", what,
        best[["setting", "n_filters", "good_retained", "bad_caught",
              "bad_rate_after", "n_same"]].to_string(index=False))
    report.figure(name, _tradeoff_figure(
        table, own, f"constraint settings, {what} (n={len(labelled)})"))


def get_validated_filters(project_path, metric_specs, predicted_run=None,
                          annotation_run=None, label_metric=None,
                          part=DEFAULT_PART, bad_labels=None,
                          max_fpr=DEFAULT_MAX_FPR, min_precision=None,
                          defaults=None, visualize=True, subset=None,
                          label_part=None, min_labelled=DEFAULT_MIN_LABELLED,
                          audit_subset=None, good_labels=None, drop_redundant=False,
                          max_total_fpr=None):
    """
    Calibrate candidate filters against human labels and return the filters an
    export should run with.

    Each candidate is swept, scored against the labels, and given the
    highest-recall rule satisfying the constraint; one that catches no bad label
    is left out. The result goes straight into `export_metrics(filters=...)`.
    What the chosen filters do together on these labels is logged.

    - `project_path` -- project to read from.
    - `metric_specs` -- the candidates, by bare metric name:
      `{"blur_variance": "below", ...}` says which side means "flag this
      one"; `"category"` instead marks a categorical metric such as
      `"cluster__cluster_id"`, whose worst categories are dropped. A list
      `["edge_fraction", ...]` works for metrics `metrics.quality` knows the
      direction of. Candidates from several runs go in one call as
      `{run: specs}`, with `predicted_run` left out.
    - `predicted_run` -- run the candidates were computed under, unless
      `metric_specs` is keyed by run.
    - `annotation_run` -- run the human labels were recorded under.
    - `label_metric` -- metric holding those labels; required. A dict-valued
      one needs the key too, as `"<metric>__<key>"`.
    - `part` -- part the candidates were measured on.
    - `bad_labels` -- label values that count as "should have been filtered".
      This or `good_labels` is required.
    - `max_fpr`, `min_precision` -- the constraint each candidate must
      satisfy on its own. `max_fpr` has a default because an unconstrained
      sweep always picks the most aggressive rule available.
    - `defaults` -- `{metric_name: threshold}` to fall back on where no
      cutoff satisfies the constraint. A metric with no fallback and no
      satisfying cutoff is left out and warned about.
    - `visualize` -- True (default): one pipeline figure per candidate, one of
      the labels before and after all the filters, and image grids of the
      labelled items (`validation.filter_grids`): by outcome, the bad ones
      kept, the good ones removed, what each filter alone removes, a strip
      of items ordered by each score with its cutoff, and each category's
      members. Also `tradeoffs`: what every setting in a grid of `max_fpr`
      and `min_precision` values would have kept and removed, with this
      call's marked, and the same on `audit_subset` where one is given.
      False writes nothing.
    - `subset` -- calibrate on this named subset's labels only. Every
      labelled occurrence if None.
    - `label_part` -- part the labels were recorded for, where it differs
      from `part`, e.g. an organism-level score calibrated against an
      abdomen's quality label. `part` if None.
    - `min_labelled` -- a category with fewer labels than this is never
      dropped.
    - `audit_subset` -- also score the chosen filters on this subset's
      labels with `audit_filters`. Needs `subset`, and must share no
      occurrence with it.
    - `good_labels` -- instead of `bad_labels`: the label values that are fine,
      every other label counting as bad. For a vocabulary of your own
      (`exclusive_label_annotation`), where a label left off `bad_labels`
      would count as clean.
    - `drop_redundant` -- drop each chosen filter that catches no bad label
      the others miss, costliest first. Candidates are calibrated one at a
      time, so two correlated scores are otherwise both kept and the second
      only costs good rows. Off by default.
    - `max_total_fpr` -- the share of good rows the filters may cost
      together. Each candidate has its own budget, so the set can exceed
      any one of them; past this the call warns and names the costliest
      filters. It changes nothing.

    Returns {export column: rule}: `(comparator, threshold)` for a continuous
    metric, `("not in", [categories])` for a categorical one. Empty if nothing
    could be calibrated, which export_metrics reads as "no filtering" -- check
    the log before trusting an unfiltered export.
    """
    if annotation_run is None:
        raise ValueError("get_validated_filters needs annotation_run=, the run "
                         "holding the human labels")
    if label_metric is None:
        raise ValueError("get_validated_filters needs label_metric=, the metric "
                         "holding the human labels")
    _require_one_label_set(bad_labels, good_labels)
    by_run = _resolve_runs(metric_specs, predicted_run)
    for specs in by_run.values():
        for metric_name, direction in specs.items():
            if direction not in DIRECTIONS:
                raise ValueError(
                    f"metric_specs['{metric_name}'] must be one of {DIRECTIONS}, "
                    f"got {direction!r}")

    if audit_subset is not None:
        if subset is None:
            raise ValueError(
                "audit_subset= needs subset=: with no calibration subset, every "
                "label is calibrated on and none is left to audit")
        shared = (set(select_ids(project_path, subset=subset))
                  & set(select_ids(project_path, subset=audit_subset)))
        if shared:
            raise ValueError(
                f"subsets '{subset}' and '{audit_subset}' share {len(shared)} "
                "occurrence(s) -- an audit has to be on labels the filters "
                "were not chosen on")

    label_part = part if label_part is None else label_part
    everything = metrics_wide(project_path, run_names=[*by_run, annotation_run],
                              parts=sorted({part, label_part}))
    df = everything
    if subset is not None and not df.empty:
        df = df[df[ID_COL].isin(select_ids(project_path, subset=subset))]
    if df.empty:
        logger.warning("no stored values for runs %s/'%s' -- no filters",
                       sorted(by_run), annotation_run)
        return {}

    label_col = column_name(annotation_run, label_part, label_metric)
    given_bad_labels = bad_labels
    bad_labels = _resolve_bad_labels(
        df[label_col] if label_col in df.columns else pd.Series(dtype=object),
        bad_labels, good_labels)
    has_category = any(direction == "category"
                       for specs in by_run.values() for direction in specs.values())

    identity = {
        "kind": "validated_filters",
        "metric_specs": by_run[predicted_run] if predicted_run is not None else by_run,
        "predicted_run": predicted_run,
        "annotation_run": annotation_run,
        "label_metric": label_metric,
        "part": part,
        "bad_labels": list(bad_labels),
        "max_fpr": max_fpr,
        "min_precision": min_precision,
        "defaults": defaults,
        # Recorded only when set, so a report written before these existed
        # keeps the filename it was written under.
        **({"subset": subset} if subset is not None else {}),
        **({"label_part": label_part} if label_part != part else {}),
        **({"min_labelled": min_labelled} if has_category else {}),
        **({"good_labels": sorted(good_labels)} if good_labels is not None else {}),
        **({"drop_redundant": True} if drop_redundant else {}),
        **({"max_total_fpr": max_total_fpr} if max_total_fpr is not None else {}),
    }
    report = pipeline_visualization.open_report(
        project_path, f"filters__{'+'.join(by_run)}__vs__{annotation_run}",
        hash_spec(identity), part=part, visualize=visualize,
        identity=identity).begin([])

    filters = {}
    continuous, categorical = {}, {}      # {column: figure name}, for the image grids
    swept = []                            # (column, metric name, direction, sweep)
    for run, specs in by_run.items():
        for metric_name, direction in specs.items():
            metric_col = column_name(run, part, metric_name)
            label = metric_name if predicted_run is not None else f"{run}__{metric_name}"
            if direction == "category":
                rule, sweep = _calibrate_category(
                    df, metric_col, label, label_col, bad_labels, max_fpr,
                    min_precision, min_labelled, report)
                categorical[metric_col] = label
            else:
                rule, sweep = _calibrate_threshold(
                    df, metric_col, metric_name, label, direction, label_col,
                    bad_labels, max_fpr, min_precision, defaults, report)
                continuous[metric_col] = label
            if rule is not None:
                filters[metric_col] = rule
            if not sweep.empty:
                swept.append((metric_col, metric_name, direction, sweep))

    # Each candidate was held to the constraint alone; together they can cost
    # more, and only the labels all of them were chosen on can say how much.
    labelled = df[df[label_col].notna()] if label_col in df.columns else df.iloc[0:0]
    if filters and not labelled.empty:
        if drop_redundant:
            filters = _drop_redundant(labelled, filters, label_col, bad_labels)
        together, kept, _stages, passes = _score_filters(labelled, filters, label_col,
                                                         bad_labels)
        _log_score(together, f"{len(filters)} filter(s) together, on the labels "
                             "they were chosen on", bad_labels)
        cost = 1 - together["good_retained"]
        if max_total_fpr is not None and cost > max_total_fpr:
            costliest = together["filters"].sort_values(
                "only_here_good", ascending=False).head(3)
            logger.warning(
                "together the filters remove %.1f%% of good rows, over "
                "max_total_fpr=%.1f%% -- the ones costing the most good rows "
                "nothing else removes:\n%s", 100 * cost, 100 * max_total_fpr,
                costliest[["filter", "only_here_bad", "only_here_good"]].to_string(index=False))
        if report:
            report.figure("together", _labels_figure(labelled[label_col], kept, together))
            draw_filter_grids(
                report, project_path, label_part, labelled, label_col, bad_labels,
                filters, kept, passes,
                strips={column: name for column, name in continuous.items()
                        if column in filters},
                categories={
                    column: (name,
                             _sweep_categories(df, column, label_col, bad_labels=bad_labels,
                                               min_labelled=min_labelled)[1],
                             filters[column][1] if column in filters else [])
                    for column, name in categorical.items()})

    # The sweeps are already made, so what any other constraint would have
    # chosen is a lookup: show the settings side by side instead of one per run.
    if report and swept and not labelled.empty:
        own = (max_fpr, min_precision)
        combinations = _tradeoff_filters(swept, labelled, label_col, bad_labels,
                                         defaults, drop_redundant, own)
        _draw_tradeoffs(report, combinations, labelled, label_col, bad_labels, own,
                        "tradeoffs", "on the calibration labels")
        if audit_subset is not None and label_col in everything.columns:
            held_out = everything[
                everything[ID_COL].isin(select_ids(project_path, subset=audit_subset))
                & everything[label_col].notna()]
            if not held_out.empty:
                _draw_tradeoffs(
                    report, combinations, held_out, label_col,
                    _resolve_bad_labels(held_out[label_col], given_bad_labels, good_labels),
                    own, "tradeoffs_audit", f"on held-out '{audit_subset}'")

    report.close()

    if audit_subset is not None and filters:
        audit_filters(project_path, filters, annotation_run, label_metric=label_metric,
                      part=label_part, bad_labels=given_bad_labels,
                      good_labels=good_labels, subset=audit_subset,
                      visualize=visualize, run_names=[*by_run, annotation_run])
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

    choice, reason = _pick(sweep, max_fpr, min_precision)
    if reason == "none satisfy":
        logger.info("no threshold satisfies min_precision=%s max_fpr=%s",
                    min_precision, max_fpr)
    elif reason == "no bad labels":
        # Recall is undefined everywhere, so there is nothing to maximize and
        # no cutoff these labels can justify.
        logger.info("no bad labels to catch -- no threshold to suggest")
    return choice


def _wilson(successes, n, z=1.96):
    """The Wilson score interval for `successes` of `n`, as (low, high); NaNs when n is 0."""
    if not n:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(result, name, successes, n):
    """Store one proportion and its interval on `result` as `<name>` and `<name>_ci`."""
    result[name] = (successes / n) if n else float("nan")
    result[f"{name}_ci"] = _wilson(successes, n)


def _score_filters(labelled, filters, label_col, bad_labels):
    """
    What a filter set does to labelled rows: the one definition of every rate
    `audit_filters` returns, shared with the summary `get_validated_filters` logs.

    - `labelled` -- wide frame of rows that have a label, holding every
      column a filter names.
    - `filters` -- as `export_metrics(filters=...)`.
    - `label_col` -- column of human labels.
    - `bad_labels` -- label values that count as "should have been filtered".

    Returns `(result, kept, stages, passes)`: the dict `audit_filters` documents,
    the boolean keep-mask over `labelled`, the row count after each filter in
    turn, and `{column: boolean mask}` of the rows passing each filter alone.
    """
    ids = labelled[ID_COL].astype(str)
    is_bad = labelled[label_col].isin(bad_labels).to_numpy()
    passes = {column: labelled.index.isin(_apply_filters(labelled, {column: rule}).index)
              for column, rule in filters.items()}
    kept = np.ones(len(labelled), bool)
    stages = {"labelled": len(labelled)}
    for column, passed in passes.items():
        kept &= passed
        stages[f"after {column}"] = int(kept.sum())

    n = len(labelled)
    n_bad = int(is_bad.sum())
    n_kept = int(kept.sum())
    n_kept_bad = int((kept & is_bad).sum())

    result = {"n": n, "n_bad": n_bad, "n_kept": n_kept, "n_kept_bad": n_kept_bad}
    _rate(result, "coverage", n_kept, n)
    _rate(result, "bad_rate_before", n_bad, n)
    _rate(result, "bad_rate_after", n_kept_bad, n_kept)
    _rate(result, "good_retained", n_kept - n_kept_bad, n - n_bad)
    _rate(result, "bad_caught", n_bad - n_kept_bad, n_bad)

    for bad_label in bad_labels:
        in_flag = (labelled[label_col] == bad_label).to_numpy()
        n_flag = int(in_flag.sum())
        result[f"recall_{bad_label}"] = (int((in_flag & ~kept).sum()) / n_flag) \
            if n_flag else float("nan")
        result[f"n_{bad_label}"] = n_flag

    rows = []
    for column, passed in passes.items():
        others = np.ones(n, bool)
        for other, other_passed in passes.items():
            if other != column:
                others &= other_passed
        rows.append({
            "filter": column,
            "removed": int((~passed).sum()),
            "removed_bad": int((~passed & is_bad).sum()),
            "removed_good": int((~passed & ~is_bad).sum()),
            "removed_only_here": int((~passed & others).sum()),
            "only_here_bad": int((~passed & others & is_bad).sum()),
            "only_here_good": int((~passed & others & ~is_bad).sum()),
        })
    result["filters"] = pd.DataFrame(
        rows, columns=["filter", "removed", "removed_bad", "removed_good",
                       "removed_only_here", "only_here_bad", "only_here_good"])
    result["kept_bad"] = sorted(ids[kept & is_bad])
    result["dropped_good"] = sorted(ids[~kept & ~is_bad])
    return result, kept, stages, passes


def _drop_redundant(labelled, filters, label_col, bad_labels, quiet=False):
    """
    `filters` without the ones that catch no bad label the others miss.

    One at a time, costliest first, re-scoring after each: of two filters that
    duplicate each other the first goes and the second, now alone in removing
    those rows, stays. A dropped filter removed no bad row uniquely, so the bad
    rows caught are unchanged and the good rows kept can only rise.
    """
    filters = dict(filters)
    while len(filters) > 1:
        table = _score_filters(labelled, filters, label_col, bad_labels)[0]["filters"]
        redundant = table[table["only_here_bad"] == 0]
        if redundant.empty:
            break
        worst = redundant.sort_values(["only_here_good", "removed_good", "filter"],
                                      ascending=[False, False, True]).iloc[0]
        if not quiet:
            logger.info("%s: dropped as redundant -- every bad label it removes is "
                        "removed by another filter, and it alone removes %d good one(s)",
                        worst["filter"], worst["only_here_good"])
        del filters[worst["filter"]]
    return filters


def _log_score(result, what, bad_labels):
    """Log one `_score_filters` result: the rates, what was removed per reason, and each filter's share."""
    logger.info(
        "%s (n=%d, %d bad): keep %.1f%% of rows; bad rate %.1f%% -> %.1f%% "
        "(95%% CI %.1f-%.1f%%); %.1f%% of good rows kept, %.1f%% of bad rows "
        "removed", what, result["n"], result["n_bad"],
        100 * result["coverage"], 100 * result["bad_rate_before"],
        100 * result["bad_rate_after"], 100 * result["bad_rate_after_ci"][0],
        100 * result["bad_rate_after_ci"][1], 100 * result["good_retained"],
        100 * result["bad_caught"])
    reasons = {label: f"{result[f'recall_{label}']:.0%} of {result[f'n_{label}']}"
               for label in bad_labels if result[f"n_{label}"]}
    if reasons:
        logger.info("removed per reason: %s", reasons)
    if len(result["filters"]) > 1:
        logger.info("per filter:\n%s", result["filters"].to_string(index=False))


def _labels_figure(labels, kept, result):
    """The labels before and after a filter set, as stacked bars."""
    labels = labels.astype(str)
    return figures.bar_chart(
        {"before": labels.value_counts().to_dict(),
         "after": labels[kept].value_counts().to_dict()},
        ylabel="labelled rows",
        title=f"bad rate {100 * result['bad_rate_before']:.1f}% -> "
              f"{100 * result['bad_rate_after']:.1f}%, "
              f"{100 * result['coverage']:.1f}% kept")


def audit_filters(project_path, filters, annotation_run,
                  label_metric=None, part=DEFAULT_PART,
                  bad_labels=None, subset=None, units=None,
                  visualize=True, run_names=None, good_labels=None):
    """
    Score a whole export filter set against human labels: how much bad data is
    left in what it keeps, and how much good data it costs.

    Audit on labels the filters were not calibrated on, drawn at random from what
    reaches the export, or the rates describe the calibration sample. Nothing is
    persisted. `get_validated_filters(audit_subset=)` calls this for the filters
    it calibrated; call it directly for a filter set written by hand.

    - `project_path` -- project to read from.
    - `filters` -- the dict handed to `export_metrics(filters=...)`, applied
      the same way: ANDed, and a missing value never passes.
    - `annotation_run` -- run the human labels were recorded under.
    - `label_metric` -- metric holding those labels; required.
    - `part` -- part the labels were recorded for.
    - `bad_labels` -- label values that count as "should have been filtered".
      This or `good_labels` is required.
    - `subset` -- audit this named subset's labels only; every labelled
      occurrence if None.
    - `units` -- as in `export_metrics`, for filters written in millimetres.
    - `visualize` -- True (default): pipeline figures of the labelled rows
      through each filter in turn and of the labels before and after, and
      image grids of the labelled items by outcome, of the bad ones kept, the
      good ones removed, and what each filter alone removes. False writes
      nothing.
    - `run_names` -- runs to read, as in `export_metrics`; every run if None.
      Must cover `annotation_run` and every run a filter names. Worth giving
      on a large project, where reading every stored value is slow.
    - `good_labels` -- instead of `bad_labels`: the label values that are fine,
      every other label counting as bad.

    Returns a dict:

    - `n`, `n_bad`, `n_kept`, `n_kept_bad` -- the counts behind every rate.
    - `coverage` -- share of labelled rows kept.
    - `bad_rate_before`, `bad_rate_after` -- share of bad rows among all
      labelled rows, and among the kept ones.
    - `good_retained` -- share of good rows kept.
    - `bad_caught` -- share of bad rows removed.
    - `<rate>_ci` -- the Wilson 95% interval of each rate above, as (low, high).
    - `recall_<label>`, `n_<label>` -- share removed, and count, per bad label.
    - `filters` -- DataFrame, one row per filter: rows it removes on its own
      (`removed`, `removed_bad`, `removed_good`) and rows no other filter
      removes (`removed_only_here`, split into `only_here_bad` and
      `only_here_good`). A filter with no `only_here_bad` catches nothing the
      others miss, and `only_here_good` is what keeping it costs.
    - `kept_bad`, `dropped_good` -- occurrence ids of the misses, sorted.

    Empty if nothing is labelled.
    """
    if label_metric is None:
        raise ValueError("audit_filters needs label_metric=, the metric holding "
                         "the human labels")
    _require_one_label_set(bad_labels, good_labels)
    filters = dict(filters or {})
    label_col = column_name(annotation_run, part, label_metric)
    df = export_metrics(project_path, path=False, manifest=False, subset=subset,
                        drop_empty=False, units=units, transform_info=True,
                        run_names=run_names)

    labelled = df[df[label_col].notna()] if label_col in df.columns else df.iloc[0:0]
    if labelled.empty:
        logger.warning("no '%s' labels%s -- nothing to audit", label_col,
                       f" in subset '{subset}'" if subset else "")
        return {}

    bad_labels = _resolve_bad_labels(labelled[label_col], bad_labels, good_labels)
    result, kept, stages, passes = _score_filters(labelled, filters, label_col, bad_labels)
    _log_score(result, f"{len(filters)} filter(s) against '{label_col}'"
                       + (f" in subset '{subset}'" if subset else ""), bad_labels)
    if result["n"] < MIN_AUDIT_ROWS or result["n_bad"] < MIN_AUDIT_BAD:
        logger.warning("only %d labelled row(s), %d bad -- the intervals are too "
                       "wide to support a claim; label more",
                       result["n"], result["n_bad"])

    identity = {
        "kind": "audit_filters",
        "filters": _recorded_filters(filters),
        "annotation_run": annotation_run,
        "label_metric": label_metric,
        "part": part,
        "bad_labels": list(bad_labels),
        "subset": subset,
        "units": units,
        **({"good_labels": sorted(good_labels)} if good_labels is not None else {}),
    }
    with pipeline_visualization.open_report(
            project_path, f"audit_filters__{annotation_run}", hash_spec(identity),
            part=part, visualize=visualize, identity=identity).begin([]) as report:
        if report:
            report.figure("funnel", figures.funnel(
                stages, title=f"labelled rows through each filter (n={result['n']})"))
            report.figure("labels", _labels_figure(labelled[label_col], kept, result))
            draw_filter_grids(report, project_path, part, labelled, label_col,
                              bad_labels, filters, kept, passes)
    return result
