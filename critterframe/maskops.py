"""Mask arithmetic with no project attached: agreement, bounds, cleanup."""

import cv2
import numpy as np


def pad_to_common_shape(mask, reference):
    """Return two boolean masks padded to their union shape.

    Args:
        mask: Boolean array.
        reference: Boolean array.
    """
    mask = np.asarray(mask) > 0
    reference = np.asarray(reference) > 0
    if mask.shape == reference.shape:
        return mask, reference

    height = max(mask.shape[0], reference.shape[0])
    width = max(mask.shape[1], reference.shape[1])

    def pad(array):
        padded = np.zeros((height, width), dtype=bool)
        padded[: array.shape[0], : array.shape[1]] = array
        return padded

    return pad(mask), pad(reference)


def mask_iou(mask, reference):
    """Return the intersection over union of two boolean masks.

    Masks of different shapes are padded to a common one. Two empty masks score 1.0.

    Args:
        mask: Boolean array.
        reference: Boolean array.
    """
    mask, reference = pad_to_common_shape(mask, reference)
    union = int((mask | reference).sum())
    return (int((mask & reference).sum()) / union) if union else 1.0


def mask_coverage(mask, reference):
    """Return the fraction of `reference`'s area that `mask` also covers.

    Area in `mask` outside `reference` costs nothing, unlike `mask_iou`.

    Args:
        mask: Boolean array.
        reference: Boolean array.
    """
    mask, reference = pad_to_common_shape(mask, reference)
    area = int(reference.sum())
    return (int((mask & reference).sum()) / area) if area else 1.0


def mask_bounds(mask):
    """Return the mask's bounding box as `{"x", "y", "width", "height"}`.

    Args:
        mask: Boolean array.

    Raises:
        ValueError: If the mask is empty.
    """
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


def largest_component(mask):
    """Return the mask's largest connected component.

    Args:
        mask: Boolean array.
    """
    mask = np.asarray(mask) > 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask

    # Row 0 is the background, so the biggest FOREGROUND component is the
    # largest area among the rest.
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def edge_distance(mask):
    """Return each mask pixel's distance to the nearest pixel outside the mask.

    The frame's edge counts as outside.

    Args:
        mask: Boolean array.

    Returns:
        A float32 array shaped like `mask`, 0 outside it.
    """
    padded = cv2.copyMakeBorder(np.asarray(mask).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def inscribed_radius(mask):
    """Return the radius of the largest disc that fits inside the mask.

    Args:
        mask: Boolean array.
    """
    return float(edge_distance(mask).max()) if np.asarray(mask).any() else 0.0
