"""Position metrics, in original image coordinates: centroid, relative_position, image_bounds."""

import cv2
import numpy as np

from ..maskops import mask_bounds
from ..core.recipes import Metric
from ..visualization.panels import annotate, overlay_mask


def centroid(name=None, unit="px"):
    """Metric: the mask's center of mass in original image coordinates, as `{"x", "y"}`."""
    return Metric("centroid", _centroid, version="1", unit=unit, metric_name=name)


def relative_position(name=None, unit="fraction"):
    """Metric: the centroid as a fraction of the image's size, as `{"x", "y"}` in 0 to 1."""
    return Metric("relative_position", _relative_position, version="1", unit=unit, metric_name=name)


def image_bounds(name=None, unit="px"):
    """Metric: the mask's bounding box in original image coordinates, as `{"x", "y", "width", "height"}`."""
    return Metric("image_bounds", _image_bounds, version="1", unit=unit, metric_name=name)


def _original_mask(segment):
    """Return the segment's mask in original image coordinates, raising if it is empty."""
    mask = segment.mask_in_original_coordinates()
    if not mask.any():
        raise ValueError("empty mask")
    return mask


def _centroid(segment):
    mask = _original_mask(segment)
    ys, xs = np.nonzero(mask)
    result = {"x": float(xs.mean()), "y": float(ys.mean())}
    _visualize(segment, mask, result, "centroid")
    return result


def _relative_position(segment):
    mask = _original_mask(segment)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    return {"x": float(xs.mean() / width), "y": float(ys.mean() / height)}


def _image_bounds(segment):
    return mask_bounds(_original_mask(segment))


def _visualize(segment, mask, position, subdir):
    """Emit the mask drawn on the original image with the reported point marked."""
    if segment.panel_sink is None:
        return

    height, width = mask.shape
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel = overlay_mask(panel, mask, color=(255, 255, 255), alpha=0.8)
    cv2.drawMarker(panel, (int(position["x"]), int(position["y"])), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    annotate(panel, f"({position['x']:.0f}, {position['y']:.0f}) of {width}x{height}")

    segment.emit_panel(panel, subdir)
