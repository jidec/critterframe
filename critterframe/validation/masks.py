"""
Compare predicted masks against reference masks.

Validation is comparison, not a separate processing system. A reference mask is
made by a segmentation recipe (a human drawing or correcting one -- see
segmentation.manual) and written to the reference mask table; a predicted mask
is made by a model recipe and written to the canonical one. Both are masks of
the same occurrence-part in the same coordinate system, so validating them is
just measuring their agreement.

That coordinate system is the reason this module is short. Persisted masks are
always in the coordinates of the original analysis image, whatever crops and
rotations the recipe that produced them went through, so a mask drawn by hand
on the full image and a mask found by a model inside a rotated sub-crop overlay
directly with no alignment step.

Nothing is written. This is a live comparison and a printed summary, cheap to
rerun every time the segmenter changes -- which is what you want, since the
reference masks only change when a human redoes one, so this always measures
"how well does TODAY's segmenter match the frozen human answer".
"""

import logging

import numpy as np
import pandas as pd

from ..recipes import DEFAULT_PART, Segment
from ..records import masks as mask_records
from ..storage.imagestore import ImageStore
from ..visualization.panels import annotate, diff_panel, side_by_side

logger = logging.getLogger(__name__)

# IoU below this is called out in a diff visualization's text overlay -- purely
# cosmetic, and it affects no comparison logic.
LOW_IOU_WARN = 0.7


def mask_iou(mask, reference):
    """
    Intersection over union of two boolean masks, padded to a common shape
    first.

    The padding matters: two masks of the same occurrence should be the same
    size, since both are in original image coordinates, but a project whose
    images were re-ingested at a different resolution could break that
    silently. Padding to the union of the two shapes makes such a mismatch show
    up as a bad IoU rather than a crash or, worse, a wrong number from a
    truncated comparison.
    """
    mask = np.asarray(mask) > 0
    reference = np.asarray(reference) > 0

    if mask.shape != reference.shape:
        height = max(mask.shape[0], reference.shape[0])
        width = max(mask.shape[1], reference.shape[1])

        def pad(array):
            padded = np.zeros((height, width), dtype=bool)
            padded[:array.shape[0], :array.shape[1]] = array
            return padded

        mask, reference = pad(mask), pad(reference)

    union = int((mask | reference).sum())
    return (int((mask & reference).sum()) / union) if union else 1.0, mask, reference


def validate_masks(project_path, part=DEFAULT_PART, transforms=(), limit=None,
                   show_worst=5, visualize=False):
    """
    Compare the canonical masks against the reference masks, per occurrence, by
    IoU.

    The comparison population is wherever a reference mask exists, so choosing
    where to make reference masks IS choosing what this measures -- there are no
    filter arguments here and there shouldn't be. On a project that screened
    before annotating (see segmentation.manual.correct_mask), reference masks
    exist only for crops a human called usable, so what comes back is "IoU over
    the crops a human called usable". Narrower than "IoU", and worth naming as
    such when reporting it -- it's also the number that describes the delivered
    dataset, since an export filtered on the same QC labels covers about the
    same population.

    project_path -- project to validate.
    part         -- part to compare.
    transforms   -- optional transforms applied to BOTH masks before comparing.
                    Use this when the reference masks represent something other
                    than raw model output -- if the human correction also erased
                    appendages, comparing raw masks counts every appendage they
                    correctly removed as a disagreement, so compare with
                    remove_appendages() applied to both instead. Applying to
                    both is the point: a transform is being held constant, not
                    tested.
    limit        -- optional cap on how many occurrences to compare.
    show_worst   -- how many of the lowest-IoU occurrences to log by id, so you
                    can go straight to reviewing real mistakes rather than
                    reading an aggregate and wondering which ones they were. 0
                    disables.
    visualize    -- save a diff panel per compared occurrence: the image
                    alongside white where the two agree, yellow where only the
                    prediction covers, red where only the reference does. One
                    image per occurrence, so pair with limit.

    Returns a DataFrame indexed by occurrence_id with an `iou` column.
    """
    transforms = list(transforms)

    predicted = mask_records.mask_lookup(project_path, part=part)
    reference = mask_records.mask_lookup(project_path, part=part, reference=True)

    occurrence_ids = sorted(set(predicted) & set(reference))
    missing = len(reference) - len(occurrence_ids)
    if limit is not None:
        occurrence_ids = occurrence_ids[:limit]

    if missing:
        logger.warning("%d reference mask(s) have no canonical mask to compare "
                       "against -- segment those occurrences first", missing)

    needs_images = bool(transforms) or visualize
    rows = []

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in occurrence_ids:
            try:
                mask = mask_records.decode_mask(predicted[occurrence_id])
                gt = mask_records.decode_mask(reference[occurrence_id])

                image = images.get(occurrence_id) if needs_images else None
                if transforms:
                    if image is None:
                        raise ValueError("no image in the image store")
                    mask = _apply(project_path, image, mask, occurrence_id, part,
                                  transforms)
                    gt = _apply(project_path, image, gt, occurrence_id, part,
                                transforms)

                iou, mask, gt = mask_iou(mask, gt)

                if visualize:
                    _visualize(project_path, image, mask, gt, iou, occurrence_id, part)

                rows.append({"occurrence_id": occurrence_id, "iou": iou})

            except Exception as exc:
                logger.warning("mask comparison failed for %s: %s",
                               occurrence_id, exc)

    df = (pd.DataFrame(rows).set_index("occurrence_id") if rows
          else pd.DataFrame(columns=["iou"]).rename_axis("occurrence_id"))

    _log_summary(df, part, show_worst)
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


def _visualize(project_path, image, mask, gt, iou, occurrence_id, part):
    """The image beside a colour-coded agreement panel."""
    from ..visualization.panels import save_panel

    panel = diff_panel(mask, gt)
    annotate(panel, f"iou {iou:.2f}{'  LOW IOU' if iou < LOW_IOU_WARN else ''}")
    annotate(panel, "white=agree  yellow=predicted only  red=reference only", line=1)

    if image is not None and image.shape[:2] == panel.shape[:2]:
        panel = side_by_side(image, panel)

    save_panel(project_path, panel, f"{occurrence_id}_{part}_iou{iou:.2f}",
                       subdir="validate_masks")


def _log_summary(df, part, show_worst):
    """Distribution summary plus the worst offenders by id."""
    if df.empty:
        logger.info("compared 0 masks for part '%s'", part)
        return

    iou = df["iou"]
    logger.info("compared %d '%s' masks to reference: mean=%.3f median=%.3f "
                "min=%.3f max=%.3f std=%.3f", len(df), part, iou.mean(),
                iou.median(), iou.min(), iou.max(),
                iou.std() if len(df) > 1 else 0.0)

    for threshold in (0.5, 0.7, 0.9):
        below = int((iou < threshold).sum())
        logger.info("  iou < %.1f: %d/%d (%.1f%%)", threshold, below, len(df),
                    100 * below / len(df))

    if show_worst:
        worst = iou.sort_values().head(show_worst)
        logger.info("  worst %d: %s", len(worst),
                    ", ".join(f"{occ}={value:.3f}" for occ, value in worst.items()))
