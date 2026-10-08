"""compare_metrics(): predicted metric values against reference values. Persists nothing."""

import contextlib
import logging

import numpy as np
import pandas as pd

from .. import selectionhelpers
from ..export import column_name, metrics_wide
from ..project import paths
from ..recipes import DEFAULT_PART, hash_spec
from ..records import masks as mask_records
from ..records import runs as run_records
from ..records.occurrences import ID_COL
from ..storage.imagestore import ImageStore
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import segment_panel

logger = logging.getLogger(__name__)


def _pair_metrics(predicted_run, reference_run, part, metric_names, columns):
    """Return `metric_names` as an ordered list of `(predicted metric, reference metric)` pairs.

    Args:
        predicted_run: Run holding the predicted values.
        reference_run: Run holding the reference values.
        part: Part compared.
        metric_names: None, a list, or a `{predicted: reference}` dict.
        columns: The wide frame's column names, to find what the two runs share.
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
        return {column[len(prefix) :] for column in columns if column.startswith(prefix)}

    shared = sorted(suffixes(predicted_run) & suffixes(reference_run))
    return [(metric_name, metric_name) for metric_name in shared]


def compare_metrics(
    project_path,
    predicted_run,
    reference_run,
    metric_names=None,
    part=DEFAULT_PART,
    current_only=True,
    show_worst=5,
    visualize=True,
):
    """Join two runs' values per occurrence and report how far apart they are.

    Args:
        project_path: Project to read from.
        predicted_run: Run name holding the automated values.
        reference_run: Run name holding the reference values.
        metric_names: None compares every metric the runs share; a list compares those
            names on both sides; a dict pairs `{predicted metric: reference metric}`, e.g.
            `{"body_length": "click_two_points__length_px"}`.
        part: Part to compare.
        current_only: Ignore values measured from masks since replaced.
        show_worst: How many of the largest percent disagreements to log by id; 0 for none.
        visualize: True, an int, or ids: a scatter per pair and a pipeline grid of the
            largest disagreements. False writes nothing.

    Returns:
        A DataFrame with one row per pair: `metric`, `reference_metric`, `n` (occurrences
        with both values), `mean_abs_diff`, `mean_pct_diff` and `median_pct_diff` (relative
        to the reference, excluding references of 0), `bias` (mean of predicted minus
        reference) and `correlation` (Pearson r).
    """
    # metrics_wide rather than a pivot of its own: newest-wins, currency, and
    # the splitting of dict values into one column per key are all decisions
    # about presenting stored values, they already live there, and a second
    # implementation of them here would be free to disagree with the export
    # about what a project currently says.
    wide = metrics_wide(
        project_path, run_names=[predicted_run, reference_run], parts=[part], current_only=current_only
    )
    if wide.empty:
        logger.warning(
            "no values to compare for runs '%s'/'%s' on part '%s'", predicted_run, reference_run, part
        )
        return pd.DataFrame()

    wide = wide.set_index(ID_COL)
    pairs = _pair_metrics(predicted_run, reference_run, part, metric_names, wide.columns)
    if not pairs:
        logger.warning("runs '%s' and '%s' share no metric names", predicted_run, reference_run)
        return pd.DataFrame()

    logger.info("agreement on part '%s': %s vs %s", part, predicted_run, reference_run)

    rows = []
    scatters = {}
    disagreements = []
    for predicted_metric, reference_metric in pairs:
        predicted_column = column_name(predicted_run, part, predicted_metric)
        reference_column = column_name(reference_run, part, reference_metric)
        label = (
            predicted_metric
            if predicted_metric == reference_metric
            else f"{predicted_metric} vs {reference_metric}"
        )

        # A column compared against itself measures nothing, and selecting it
        # twice gives a duplicated-column frame in which every statistic below
        # degenerates into an unreadable pandas TypeError. Reached by comparing
        # a run with itself, which is a plausible typo.
        if predicted_column == reference_column:
            logger.info("  %s: predicted and reference are the same column, skipping", label)
            continue

        missing = [column for column in (predicted_column, reference_column) if column not in wide.columns]
        if missing:
            logger.info("  %s: %s not measured, skipping", label, ", ".join(missing))
            continue

        pair = wide[[predicted_column, reference_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            logger.info("  %s: no occurrences with both values (or not numeric)", label)
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
            logger.warning(
                "  %s: %d occurrence(s) have a reference of 0; "
                "counted in n and bias, excluded from the percent "
                "figures",
                label,
                excluded,
            )

        row = {
            "metric": predicted_metric,
            "reference_metric": reference_metric,
            "n": len(pair),
            "mean_abs_diff": float(difference.abs().mean()),
            "mean_pct_diff": float(percent.mean()),
            "median_pct_diff": float(percent.median()),
            "bias": float(difference.mean()),
            "correlation": float(pair[predicted_column].corr(pair[reference_column]))
            if len(pair) > 1
            else float("nan"),
        }
        rows.append(row)

        logger.info(
            "  %s: n=%d  mean|diff|=%.2f  mean|%%diff|=%.1f%%  median|%%diff|=%.1f%%  bias=%+.2f  r=%.3f",
            label,
            row["n"],
            row["mean_abs_diff"],
            row["mean_pct_diff"],
            row["median_pct_diff"],
            row["bias"],
            row["correlation"],
        )

        if show_worst:
            # Ranked on the absolute value (worst_n drops NaN itself, same as
            # the dropna() this replaces), displayed with its sign -- the
            # signed and unsigned series diverge here, so only the ranking
            # step is shared.
            worst = selectionhelpers.worst_n(percent, show_worst, ascending=False)
            if worst:
                logger.info(
                    "    worst %d: %s",
                    len(worst),
                    ", ".join(
                        f"{occurrence_id}={signed_percent[occurrence_id]:+.0f}%"
                        for occurrence_id, _value in worst
                    ),
                )

        if visualize:
            scatters[
                predicted_metric
                if predicted_metric == reference_metric
                else f"{predicted_metric}__vs__{reference_metric}"
            ] = (
                pair[reference_column].tolist(),
                pair[predicted_column].tolist(),
                reference_metric,
                predicted_metric,
            )
            for occurrence_id, value in percent.dropna().items():
                disagreements.append(
                    {
                        "occurrence_id": str(occurrence_id),
                        "abs_pct": float(value),
                        "signed_pct": float(signed_percent[occurrence_id]),
                        "predicted_metric": predicted_metric,
                        "reference_metric": reference_metric,
                        "predicted": float(pair.at[occurrence_id, predicted_column]),
                        "reference": float(pair.at[occurrence_id, reference_column]),
                    }
                )

    if visualize and rows:
        _visualize_agreement(
            project_path,
            predicted_run,
            reference_run,
            part,
            pairs,
            current_only,
            visualize,
            scatters,
            disagreements,
        )

    return pd.DataFrame(rows)


def _visualize_agreement(
    project_path, predicted_run, reference_run, part, pairs, current_only, visualize, scatters, disagreements
):
    """Write the scatter per pair and the grid of the worst disagreements."""
    pointers = run_records.current_recipe_pointers(project_path)
    identity = {
        "kind": "compare_metrics",
        "predicted_run": predicted_run,
        "reference_run": reference_run,
        "recipes": {
            predicted_run: pointers.get((predicted_run, part)),
            reference_run: pointers.get((reference_run, part)),
        },
        "part": part,
        "pairs": [list(pair) for pair in pairs],
        "current_only": current_only,
    }

    disagreements.sort(key=lambda entry: -entry["abs_pct"])
    count = pipeline_visualization.sample_count(visualize)
    if count is None:
        wanted = {str(occurrence_id) for occurrence_id in visualize}
        kept = [entry for entry in disagreements if entry["occurrence_id"] in wanted]
    else:
        kept = disagreements[:count]
    keys = [f"{entry['occurrence_id']}__{entry['predicted_metric']}" for entry in kept]

    report = pipeline_visualization.open_report(
        project_path,
        f"compare_metrics__{predicted_run}__vs__{reference_run}",
        hash_spec(identity),
        part=part,
        visualize=keys,
        identity=identity,
    ).begin(keys)

    for name, (reference_values, predicted_values, reference_metric, predicted_metric) in scatters.items():
        report.figure(
            name,
            figures.scatter(
                reference_values,
                predicted_values,
                diagonal=True,
                xlabel=f"{reference_run}: {reference_metric}",
                ylabel=f"{predicted_run}: {predicted_metric}",
                title=f"{predicted_metric} vs reference (n={len(reference_values)})",
            ),
        )

    masks = mask_records.mask_lookup(
        project_path, part=part, occurrence_ids=[entry["occurrence_id"] for entry in kept]
    )
    has_images = paths.images_path(project_path).exists()
    with contextlib.ExitStack() as stack:
        images = stack.enter_context(ImageStore(project_path, readonly=True)) if has_images and kept else None
        for key, entry in zip(keys, kept):
            try:
                image = images.get(entry["occurrence_id"]) if images is not None else None
                mask = None
                if image is None:
                    # Values without a specimen still belong on the sheet.
                    image = np.zeros((120, 240, 3), np.uint8)
                else:
                    mask_row = masks.get(entry["occurrence_id"])
                    mask = mask_records.decode_mask(mask_row) if mask_row is not None else None

                report.panel(
                    key,
                    "disagreement",
                    segment_panel(
                        image,
                        mask,
                        lines=[
                            f"{entry['predicted_metric']}={entry['predicted']:.3g}",
                            f"{entry['reference_metric']}={entry['reference']:.3g}",
                            f"diff {entry['signed_pct']:+.0f}%",
                        ],
                    ),
                )
            except Exception as exc:
                report.failure(key, exc)
            report.done(key)

    report.close()
