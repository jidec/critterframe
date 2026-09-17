"""
IoU and coverage against reference masks. Compares, persists nothing.

Masks are padded to a common shape before comparison, so two masks of one
occurrence at different resolutions show up as a bad score rather than as a
crash or a wrong number from a truncated comparison.
"""

import logging

import numpy as np
import pandas as pd

from ..recipes import DEFAULT_PART, Segment
from ..records import masks as mask_records
from ..storage.imagestore import ImageStore
from ..visualization.panels import (
    PanelFiles,
    annotate,
    diff_panel,
    save_panel,
    side_by_side,
)

logger = logging.getLogger(__name__)

# Score below this is called out in a diff visualization's text overlay --
# purely cosmetic, and it affects no comparison logic.
LOW_IOU_WARN = 0.7
LOW_COVERAGE_WARN = 0.7


def _pad_to_common_shape(mask, reference):
    """
    Pad two boolean masks to their union shape.

    Two masks of the same occurrence should be the same size, since both are
    in original image coordinates, but a project whose images were
    re-ingested at a different resolution could break that silently. Padding
    makes such a mismatch show up as a bad score rather than a crash or,
    worse, a wrong number from a truncated comparison.
    """
    mask = np.asarray(mask) > 0
    reference = np.asarray(reference) > 0

    if mask.shape == reference.shape:
        return mask, reference

    height = max(mask.shape[0], reference.shape[0])
    width = max(mask.shape[1], reference.shape[1])

    def pad(array):
        padded = np.zeros((height, width), dtype=bool)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded

    return pad(mask), pad(reference)


def mask_iou(mask, reference):
    """
    Intersection over union of two boolean masks, padded to a common shape
    first.
    """
    mask, reference = _pad_to_common_shape(mask, reference)
    union = int((mask | reference).sum())
    return (int((mask & reference).sum()) / union) if union else 1.0, mask, reference


def mask_coverage(mask, reference):
    """
    Fraction of `reference`'s area that `mask` also covers, padded to a
    common shape first.

    Unlike `mask_iou`, area in `mask` outside `reference` costs nothing. The
    metric for a predicted mask that only needs to CONTAIN a reference region
    rather than match its extent -- an organism mask that also picks up the
    wings is still a perfect answer for a part segmenter that only needs the
    body present.
    """
    mask, reference = _pad_to_common_shape(mask, reference)
    area = int(reference.sum())
    return (int((mask & reference).sum()) / area) if area else 1.0


def validate_masks(project_path, part=DEFAULT_PART, reference_part=None,
                   transforms=(), steps=None, limit=None, show_worst=5,
                   visualize=False, metric="iou", label=None):
    """
    Compare masks against the reference masks, per occurrence.

    By default the predicted side is the stored canonical masks. Pass `steps`
    to compute the predicted side live instead, so a candidate recipe can be
    checked against the reference set without a prior `run_segments()` pass.

    The comparison population is wherever a reference mask exists, so choosing
    where to make reference masks IS choosing what this measures. On a project
    that screened before annotating, references exist only for crops a human
    called usable, so the result is "IoU over the crops a human called usable" --
    narrower than "IoU", and worth naming as such when reporting it.

    - `project_path` -- project to validate.
    - `part` -- part to compare.
    - `reference_part` -- reference part to compare `part` against, if
      different, e.g. a merged part from `merge_masks()`. Defaults to `part`.
    - `transforms` -- optional transforms applied to BOTH masks before
      comparing. Use this when the reference represents something other
      than raw model output: if the human correction also erased
      appendages, comparing raw masks counts every appendage they correctly
      removed as a disagreement. Applying to both is the point -- a
      transform is held constant, not tested.
    - `steps` -- ordered operations computing the predicted mask live from
      each occurrence's image, e.g. `[segment(candidate_model)]`, in place
      of the stored canonical masks. Skips the canonical table entirely, so
      the population becomes every occurrence with a reference mask rather
      than the intersection with it. Nothing computed here is persisted.
    - `limit` -- optional cap on how many occurrences to compare.
    - `show_worst` -- how many of the lowest-scoring occurrences to log by
      id. 0 disables.
    - `visualize` -- save a diff panel per compared occurrence: white where
      the two agree, yellow where only the prediction covers, red where
      only the reference does. One image each, so pair with `limit`. With
      `steps`, also surfaces any diagnostic panel a step emits (e.g. a
      segmentation model's own `visualize()`), written alongside the diff
      panel under the same `label`.
    - `metric` -- `"iou"` (default) or `"coverage"`: the fraction of the
      reference the prediction covers, ignoring any predicted area outside
      it. Use `"coverage"` when extra predicted area shouldn't count against
      a candidate, e.g. an organism mask feeding a part segmenter that only
      needs the body present, not a tight match to the organism's extent.
    - `label` -- namespaces this call's visualization output under
      `visualizations/validate_masks/<label>/` instead of the flat
      `visualizations/validate_masks/`. Pass a distinct label per call when
      comparing several candidates -- e.g. a parameter sweep -- so one call's
      output doesn't overwrite another's.

    Returns a DataFrame indexed by occurrence_id with an `iou` or `coverage`
    column, matching `metric`.
    """
    if metric not in ("iou", "coverage"):
        raise ValueError(f"metric must be 'iou' or 'coverage', got {metric!r}")
    reference_part = part if reference_part is None else reference_part
    transforms = list(transforms)
    subdir = "validate_masks" if label is None else f"validate_masks/{label}"

    reference = mask_records.mask_lookup(project_path, part=reference_part,
                                         reference=True)

    if steps is None:
        predicted = mask_records.mask_lookup(project_path, part=part)
        occurrence_ids = sorted(set(predicted) & set(reference))
        missing = len(reference) - len(occurrence_ids)
        if missing:
            logger.warning("%d reference mask(s) have no canonical '%s' mask to "
                           "compare against -- segment those occurrences "
                           "first", missing, part)
    else:
        predicted = None
        occurrence_ids = sorted(reference)

    if limit is not None:
        occurrence_ids = occurrence_ids[:limit]

    needs_images = bool(transforms) or visualize or steps is not None
    rows = []
    panel_sink = (PanelFiles(project_path, prefix=subdir)
                  if visualize and steps is not None else None)

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in occurrence_ids:
            try:
                image = images.get(occurrence_id) if needs_images else None
                gt = mask_records.decode_mask(reference[occurrence_id])

                if steps is not None:
                    if image is None:
                        raise ValueError("no image in the image store")
                    mask = _compute(project_path, image, occurrence_id, part, steps,
                                    panel_sink=panel_sink)
                else:
                    mask = mask_records.decode_mask(predicted[occurrence_id])

                if transforms:
                    if image is None:
                        raise ValueError("no image in the image store")
                    mask = _apply(project_path, image, mask, occurrence_id, part,
                                  transforms)
                    gt = _apply(project_path, image, gt, occurrence_id, part,
                                transforms)

                if metric == "iou":
                    score, mask, gt = mask_iou(mask, gt)
                else:
                    mask, gt = _pad_to_common_shape(mask, gt)
                    score = mask_coverage(mask, gt)

                if visualize:
                    low_warn = LOW_IOU_WARN if metric == "iou" else LOW_COVERAGE_WARN
                    _visualize(project_path, image, mask, gt, score, occurrence_id,
                              part, column=metric, low_warn=low_warn, subdir=subdir)

                rows.append({"occurrence_id": occurrence_id, metric: score})

            except Exception as exc:
                logger.warning("mask comparison failed for %s: %s",
                               occurrence_id, exc)

    df = (pd.DataFrame(rows).set_index("occurrence_id") if rows
          else pd.DataFrame(columns=[metric]).rename_axis("occurrence_id"))

    _log_summary(df, part, show_worst, column=metric)
    return df


def _apply(project_path, image, mask, occurrence_id, part, transforms):
    """
    Run a transform chain over one mask and return the result. Both the
    prediction and the reference go through this, so whatever the chain does,
    it does to both.
    """
    state = Segment(image, mask=mask, occurrence_id=occurrence_id, part=part,
                    project_path=project_path)
    for operation in transforms:
        state, _info = operation(state)
    return state.mask


def _compute(project_path, image, occurrence_id, part, steps, panel_sink=None):
    """
    Run a segmentation chain over one image and return the resulting mask,
    warped back to original coordinates -- the frame reference masks are
    stored in, in case `steps` moved pixels before segmenting.
    """
    state = Segment(image, occurrence_id=occurrence_id, part=part,
                    project_path=project_path, panel_sink=panel_sink)
    for operation in steps:
        state, _info = operation(state)
    return state.mask_in_original_coordinates()


def _visualize(project_path, image, mask, gt, score, occurrence_id, part,
               column="iou", low_warn=LOW_IOU_WARN, subdir="validate_masks"):
    """The image beside a colour-coded agreement panel."""
    panel = diff_panel(mask, gt)
    annotate(panel, f"{column} {score:.2f}{'  LOW' if score < low_warn else ''}")
    annotate(panel, "white=agree  yellow=predicted only  red=reference only", line=1)

    if image is not None and image.shape[:2] == panel.shape[:2]:
        panel = side_by_side(image, panel)

    save_panel(project_path, panel, f"{occurrence_id}_{part}_{column}{score:.2f}",
                       subdir=subdir)


def _log_summary(df, part, show_worst, column="iou"):
    """Distribution summary plus the worst offenders by id."""
    if df.empty:
        logger.info("compared 0 masks for part '%s'", part)
        return

    values = df[column]
    logger.info("compared %d '%s' masks to reference: mean=%.3f median=%.3f "
                "min=%.3f max=%.3f std=%.3f", len(df), part, values.mean(),
                values.median(), values.min(), values.max(),
                values.std() if len(df) > 1 else 0.0)

    for threshold in (0.5, 0.7, 0.9):
        below = int((values < threshold).sum())
        logger.info("  %s < %.1f: %d/%d (%.1f%%)", column, threshold, below,
                    len(df), 100 * below / len(df))

    if show_worst:
        worst = values.sort_values().head(show_worst)
        logger.info("  worst %d: %s", len(worst),
                    ", ".join(f"{occ}={value:.3f}" for occ, value in worst.items()))
