"""validate_masks(): IoU or coverage against reference masks. Persists nothing."""

import logging

import pandas as pd

from .. import segments as segment_iteration
from .. import selectionhelpers
from ..maskops import mask_coverage, mask_iou, pad_to_common_shape
from ..recipes import DEFAULT_PART, hash_spec
from ..records import masks as mask_records
from ..storage.imagestore import ImageStore
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate, diff_panel, side_by_side

logger = logging.getLogger(__name__)

# Score below this is called out in a diff visualization's text overlay --
# purely cosmetic, and it affects no comparison logic.
LOW_IOU_WARN = 0.7
LOW_COVERAGE_WARN = 0.7


def validate_masks(
    project_path,
    part=DEFAULT_PART,
    parts=None,
    reference_part=None,
    transforms=(),
    steps=None,
    limit=None,
    show_worst=5,
    visualize=True,
    metric="iou",
    label=None,
):
    """Compare masks against the reference masks, per occurrence.

    The comparison covers the occurrences that have a reference mask, so where references
    were drawn decides what the result measures.

    Args:
        project_path: Project to validate.
        part: Part to compare.
        parts: Several parts, each compared the same way. Not with `reference_part`.
        reference_part: Reference part to compare `part` against; `part` if None.
        transforms: Transforms applied to both masks before comparing, for a reference
            that represents more than raw model output, e.g. appendages removed.
        steps: Operations computing the predicted mask from each image, e.g.
            `[segment(candidate_model)]`, in place of the stored canonical masks. Nothing
            computed is persisted.
        limit: Cap on the occurrences compared.
        show_worst: How many of the lowest-scoring occurrences to log by id; 0 for none.
        visualize: True, an int, or ids: a pipeline grid of the lowest-scoring occurrences
            as diff panels (white agreement, yellow prediction only, red reference only),
            and a score histogram. False writes nothing.
        metric: `"iou"`, or `"coverage"`: the fraction of the reference the prediction
            covers, ignoring predicted area outside it.
        label: Names this call's pipeline files `validate_masks__<label>`.

    Returns:
        A DataFrame indexed by occurrence id with an `iou` or `coverage` column, or
        `{part: DataFrame}` with `parts`.
    """
    if metric not in ("iou", "coverage"):
        raise ValueError(f"metric must be 'iou' or 'coverage', got {metric!r}")
    if parts and reference_part is not None:
        raise ValueError(
            "reference_part names one specific pairing, so it can't be given "
            "alongside parts= -- validate those parts one call each"
        )
    transforms = list(transforms)

    if parts:
        return {
            target_part: _validate_one_part(
                project_path,
                target_part,
                target_part,
                transforms,
                steps,
                limit,
                show_worst,
                visualize,
                metric,
                label,
            )
            for target_part in parts
        }
    return _validate_one_part(
        project_path,
        part,
        part if reference_part is None else reference_part,
        transforms,
        steps,
        limit,
        show_worst,
        visualize,
        metric,
        label,
    )


def _validate_one_part(
    project_path, part, reference_part, transforms, steps, limit, show_worst, visualize, metric, label
):
    """Compare one part against its reference."""
    reference = mask_records.mask_lookup(project_path, part=reference_part, reference=True)

    if steps is None:
        predicted = mask_records.mask_lookup(project_path, part=part)
        occurrence_ids = selectionhelpers.require_present(predicted, reference=reference)
        missing = len(reference) - len(occurrence_ids)
        if missing:
            logger.warning(
                "%d reference mask(s) have no canonical '%s' mask to "
                "compare against -- segment those occurrences "
                "first",
                missing,
                part,
            )
    else:
        predicted = None
        occurrence_ids = sorted(reference)

    if limit is not None:
        occurrence_ids = occurrence_ids[:limit]

    identity = {
        "kind": "validate_masks",
        "part": part,
        "reference_part": reference_part,
        "metric": metric,
        "transforms": [operation.spec() for operation in transforms],
        "steps": None if steps is None else [operation.spec() for operation in steps],
    }
    report = pipeline_visualization.open_report(
        project_path,
        "validate_masks" if label is None else f"validate_masks__{label}",
        hash_spec(identity),
        part=part,
        visualize=visualize,
        rank="lowest",
        identity=identity,
    ).begin(occurrence_ids)
    low_warn = LOW_IOU_WARN if metric == "iou" else LOW_COVERAGE_WARN

    needs_images = bool(transforms) or bool(report) or steps is not None
    rows = []

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in occurrence_ids:
            score = None
            try:
                image = images.get(occurrence_id) if needs_images else None
                gt = mask_records.decode_mask(reference[occurrence_id])

                if steps is not None:
                    if image is None:
                        raise ValueError("no image in the image store")
                    mask = _compute(
                        project_path, image, occurrence_id, part, steps, panel_sink=report.sink(occurrence_id)
                    )
                else:
                    mask = mask_records.decode_mask(predicted[occurrence_id])

                if transforms:
                    if image is None:
                        raise ValueError("no image in the image store")
                    mask = _apply(project_path, image, mask, occurrence_id, part, transforms)
                    gt = _apply(project_path, image, gt, occurrence_id, part, transforms)

                # Padded here as well as inside the measure, so the diff
                # panel draws the same two arrays the score was computed from.
                mask, gt = pad_to_common_shape(mask, gt)
                score = mask_iou(mask, gt) if metric == "iou" else mask_coverage(mask, gt)

                if report.wants(occurrence_id):
                    report.panel(
                        occurrence_id,
                        "compare",
                        _diff_panel(image, mask, gt, score, column=metric, low_warn=low_warn),
                    )

                rows.append({"occurrence_id": occurrence_id, metric: score})

            except Exception as exc:
                report.failure(occurrence_id, exc)
                logger.warning("mask comparison failed for %s: %s", occurrence_id, exc)

            report.done(occurrence_id, rank_value=score)

    df = (
        pd.DataFrame(rows).set_index("occurrence_id")
        if rows
        else pd.DataFrame(columns=[metric]).rename_axis("occurrence_id")
    )

    if report and not df.empty:
        report.figure(
            metric,
            figures.histogram(
                df[metric].tolist(),
                bins=20,
                xlabel=metric,
                marks={"low": low_warn},
                title=f"{part} vs reference {reference_part}: {metric} (n={len(df)})",
            ),
        )
    report.close()

    _log_summary(df, part, show_worst, column=metric)
    return df


def _apply(project_path, image, mask, occurrence_id, part, transforms):
    """Run a transform chain over one mask, and return the result in original image coordinates.

    Each side's own geometry drives the chain, so a transform that moves pixels leaves the
    two in different frames; inverting each back is what keeps the comparison aligned.
    """
    state = segment_iteration.build_segment(
        image, mask=mask, occurrence_id=occurrence_id, part=part, project_path=project_path
    )
    for operation in transforms:
        state, _info = operation(state)
    return state.mask_in_original_coordinates()


def _compute(project_path, image, occurrence_id, part, steps, panel_sink=None):
    """Run a segmentation chain over one image, and return the mask in original coordinates.

    Not built on `iterate_segments`: this report ranks by score, and that loop's own
    `done()` would commit an item's panels before the score exists.
    """
    state = segment_iteration.build_segment(
        image, occurrence_id=occurrence_id, part=part, project_path=project_path, panel_sink=panel_sink
    )
    for operation in steps:
        state, _info = operation(state)
    return state.mask_in_original_coordinates()


def _diff_panel(image, mask, gt, score, column="iou", low_warn=LOW_IOU_WARN):
    """Return the image beside a color-coded agreement panel."""
    panel = diff_panel(mask, gt)
    annotate(panel, f"{column} {score:.2f}{'  LOW' if score < low_warn else ''}")
    annotate(panel, "white=agree  yellow=predicted only  red=reference only", line=1)

    if image is not None and image.shape[:2] == panel.shape[:2]:
        panel = side_by_side(image, panel)
    return panel


def _log_summary(df, part, show_worst, column="iou"):
    """Log the score distribution and the worst occurrences by id."""
    if df.empty:
        logger.info("compared 0 masks for part '%s'", part)
        return

    values = df[column]
    logger.info(
        "compared %d '%s' masks to reference: mean=%.3f median=%.3f min=%.3f max=%.3f std=%.3f",
        len(df),
        part,
        values.mean(),
        values.median(),
        values.min(),
        values.max(),
        values.std() if len(df) > 1 else 0.0,
    )

    for threshold in (0.5, 0.7, 0.9):
        below = int((values < threshold).sum())
        logger.info("  %s < %.1f: %d/%d (%.1f%%)", column, threshold, below, len(df), 100 * below / len(df))

    if show_worst:
        worst = selectionhelpers.worst_n(values, show_worst)
        logger.info("  worst %d: %s", len(worst), ", ".join(f"{occ}={value:.3f}" for occ, value in worst))
