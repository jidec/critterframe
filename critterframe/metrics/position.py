"""
Where in the image the organism is: centroid, relative_position, image_bounds.

Reported in ORIGINAL image coordinates, whatever crops a recipe applied, so two
occurrences are comparable.
"""

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import annotate, overlay_mask


def centroid(name=None, unit="px"):
    """
    Metric: the mask's centre of mass in ORIGINAL image coordinates, as
    {"x", "y"}.

    Centre of mass rather than bounding-box centre, so a long tail doesn't drag
    the reported position the way it would if half the box were empty.
    """
    return Metric("centroid", _centroid, version="1", unit=unit, metric_name=name)


def relative_position(name=None, unit="fraction"):
    """
    Metric: the centroid as a fraction of image size, as {"x", "y"} in [0, 1].

    The comparable version of centroid(): experimental boxes photographed at
    different resolutions or slightly different distances give position in
    pixels that can't be pooled, while position as a fraction of the frame can.
    """
    return Metric("relative_position", _relative_position, version="1",
                  unit=unit, metric_name=name)


def image_bounds(name=None, unit="px"):
    """
    Metric: the mask's bounding box in ORIGINAL image coordinates, as
    {"x", "y", "width", "height"}.

    The positional counterpart to dimensions.bounding_box (which reports the
    box in the working frame): this one says where the organism is in the
    picture, that one says how big it is once oriented.
    """
    return Metric("image_bounds", _image_bounds, version="1", unit=unit,
                  metric_name=name)


def _original_mask(segment):
    """The segment's mask back in original image coordinates, raising if empty."""
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
    mask = _original_mask(segment)
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


def _visualize(segment, mask, position, subdir):
    """
    The mask drawn on the ORIGINAL image with the reported point marked -- the
    check that matters here, since a position metric's failure mode is being
    correct in the wrong coordinate system, which no number on its own reveals.
    """
    if segment.panel_sink is None:
        return

    height, width = mask.shape
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel = overlay_mask(panel, mask, color=(255, 255, 255), alpha=0.8)
    cv2.drawMarker(panel, (int(position["x"]), int(position["y"])),
                   (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    annotate(panel, f"({position['x']:.0f}, {position['y']:.0f}) "
                    f"of {width}x{height}")

    segment.emit_panel(panel, subdir)
