"""
Mask arithmetic every layer needs: agreement, bounds, cleanup.

Boolean arrays in, numbers or arrays out -- no project, no records, no torch. Segmentation,
metrics, transforms, validation and the trainable segmenter all ask the same questions of a mask,
and an IoU computed five ways is five chances for two of them to disagree about the empty case.
"""

import cv2
import numpy as np


def pad_to_common_shape(mask, reference):
    """
    Two boolean masks padded to their union shape.

    Two masks of one occurrence should already match, since both are in
    original image coordinates, but a project whose images were re-ingested at
    a different resolution could break that silently. Padding makes such a
    mismatch a bad score rather than a crash, or worse, a wrong number from a
    truncated comparison.

    - `mask`, `reference` -- boolean arrays.
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
    Intersection over union of two boolean masks, padded to a common shape first.

    Two empty masks score 1.0 rather than dividing by zero: neither found
    anything, and they do not disagree about where it is.

    - `mask`, `reference` -- boolean arrays.
    """
    mask, reference = pad_to_common_shape(mask, reference)
    union = int((mask | reference).sum())
    return (int((mask & reference).sum()) / union) if union else 1.0


def mask_coverage(mask, reference):
    """
    Fraction of `reference`'s area that `mask` also covers, padded to a common shape first.

    Unlike `mask_iou`, area in `mask` outside `reference` costs nothing: the
    measure for a prediction that only needs to CONTAIN a region, e.g. an
    organism mask feeding a part segmenter that needs the body present rather
    than a tight match to the organism's extent.

    - `mask`, `reference` -- boolean arrays.
    """
    mask, reference = pad_to_common_shape(mask, reference)
    area = int(reference.sum())
    return (int((mask & reference).sum()) / area) if area else 1.0


def mask_bounds(mask):
    """
    The mask's bounding box as `{"x", "y", "width", "height"}`, in its own frame.

    Inclusive of the last row and column, so width is what a crop of that box
    would be. Raises on an empty mask, which has no box to report.

    - `mask` -- boolean array.
    """
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


def largest_component(mask):
    """
    The mask's biggest connected component, or the mask unchanged if it has at most one.

    - `mask` -- boolean array.
    """
    mask = np.asarray(mask) > 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask

    # Row 0 is the background, so the biggest FOREGROUND component is the
    # largest area among the rest.
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest
