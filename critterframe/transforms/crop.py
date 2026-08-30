"""
Framing transforms: crop, crop_to_mask, rotate, resize, remove_background.

Each composes its affine onto the segment, so a mask found in a crop still
inverts back to original image coordinates.
"""

import logging

import cv2
import numpy as np

from ..recipes import Transform
from ..visualization.panels import annotate, side_by_side
from .orient import apply_affine, rotation_matrix

logger = logging.getLogger(__name__)

# Named regions as (x fraction, y fraction, width fraction, height fraction) of
# the working frame. Fractions rather than pixels so one named region works
# across a collection whose images aren't all the same size.
REGIONS = {
    "upper_left":  (0.0, 0.0, 0.5, 0.5),
    "upper_right": (0.5, 0.0, 0.5, 0.5),
    "lower_left":  (0.0, 0.5, 0.5, 0.5),
    "lower_right": (0.5, 0.5, 0.5, 0.5),
    "top":         (0.0, 0.0, 1.0, 0.5),
    "bottom":      (0.0, 0.5, 1.0, 0.5),
    "left":        (0.0, 0.0, 0.5, 1.0),
    "right":       (0.5, 0.0, 0.5, 1.0),
    "center":      (0.25, 0.25, 0.5, 0.5),
}

# Default slack around a mask's bounding box for crop_to_mask, as a fraction of
# the box's size -- enough that a tight box doesn't clip the true edge, without
# dragging in much extra background.
BBOX_PAD_FRAC = 0.05


def crop(region=None, x=None, y=None, width=None, height=None):
    """
    Operation: crop the working frame to a region.

    Give either a named region or an explicit box, not both.

    region -- one of REGIONS ("upper_left", "center", ...), taken as fractions
              of the current frame.
    x, y, width, height -- explicit box in pixels of the current frame.
    """
    if region is not None and any(v is not None for v in (x, y, width, height)):
        raise ValueError("crop takes either region= or an explicit x/y/width/height box")
    if region is None and None in (x, y, width, height):
        raise ValueError("crop needs region=, or all of x=, y=, width=, height=")
    if region is not None and region not in REGIONS:
        raise ValueError(f"unknown region {region!r} -- expected one of {sorted(REGIONS)}")

    return Transform("crop", _crop, {
        "region": region, "x": x, "y": y, "width": width, "height": height,
    }, version="1")


def crop_to_mask(pad=BBOX_PAD_FRAC):
    """
    Operation: crop the working frame to the current mask's bounding box.

    The usual preprocessing before a part-specific model: it puts the organism
    at a consistent size in a consistent frame, so the model doesn't have to
    cope with the specimen occupying 2% of one image and 60% of the next.

    pad -- slack around the box as a fraction of its size.
    """
    return Transform("crop_to_mask", _crop_to_mask, {"pad": pad}, version="1")


def rotate(degrees):
    """
    Operation: rotate the working frame by a fixed angle, expanding the canvas
    so nothing is clipped.

    For collections whose specimens are mounted at a known angle. Use
    transforms.orient.orient() instead when the angle should be found from the
    organism rather than stated.

    degrees -- rotation in degrees, positive counter-clockwise.
    """
    return Transform("rotate", _rotate, {"degrees": degrees}, version="1")


def resize(width=None, height=None, scale=None):
    """
    Operation: resize the working frame.

    Give width and/or height in pixels, or a scale factor. With only one of
    width/height, the other follows to preserve aspect ratio -- distorting an
    organism's proportions would corrupt every shape trait measured afterward.

    width, height -- target size in pixels.
    scale         -- multiplier applied to both dimensions.
    """
    if scale is not None and (width is not None or height is not None):
        raise ValueError("resize takes either scale= or width=/height=, not both")
    if scale is None and width is None and height is None:
        raise ValueError("resize needs scale=, width=, or height=")

    return Transform("resize", _resize,
                     {"width": width, "height": height, "scale": scale},
                     version="1")


def remove_background(fill=0):
    """
    Operation: blank out every pixel outside the current mask, leaving the
    organism on a flat background.

    The standard preprocessing before a part-specific segmenter: with the
    background gone, a head/thorax/abdomen model can't learn to key off the
    substrate a specimen happened to be photographed on, and doesn't have to
    rediscover the organism boundary that whole-organism segmentation already
    established.

    Doesn't move any pixels, so the mapping back to original coordinates is
    untouched, and the mask passes through unchanged -- what changes is only
    what the image shows outside it.

    fill -- value written to background pixels; 0 (black) by default.
    """
    return Transform("remove_background", _remove_background, {"fill": fill},
                     version="1")


def _remove_background(segment, fill=0):
    """Blank the image outside the mask."""
    mask = segment.require_mask()
    image = np.asarray(segment.image).copy()
    image[~mask] = fill

    info = {"fill": fill,
            "background_fraction": float((~mask).sum() / mask.size)}

    blanked = segment.replace(image=image)
    _visualize(segment, blanked, "remove_background", info)
    return blanked, info


def _translation(x0, y0):
    """Affine mapping current coordinates to a crop starting at (x0, y0)."""
    return np.array([[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)]])


def _apply_box(segment, x0, y0, x1, y1, name):
    """
    Shared crop mechanics: clip the box to the frame, slice image and mask, and
    record the translation so the result still maps back to original
    coordinates.
    """
    height, width = segment.shape
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"{name} produced an empty box ({x0},{y0})-({x1},{y1}) in a "
            f"{width}x{height} frame"
        )

    image = np.asarray(segment.image)[y0:y1, x0:x1]
    mask = None if segment.mask is None else segment.mask[y0:y1, x0:x1]

    info = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
            "source_width": width, "source_height": height}

    cropped = segment.replace(image=image,
                              mask=False if mask is None else mask,
                              applied=_translation(x0, y0))
    _visualize(segment, cropped, name, info)
    return cropped, info


def _crop(segment, region=None, x=None, y=None, width=None, height=None):
    """Crop to a named region (fractions of the frame) or an explicit pixel box."""
    if region is not None:
        frame_height, frame_width = segment.shape
        fx, fy, fw, fh = REGIONS[region]
        x, y = fx * frame_width, fy * frame_height
        width, height = fw * frame_width, fh * frame_height

    return _apply_box(segment, x, y, x + width, y + height, "crop")


def _crop_to_mask(segment, pad=BBOX_PAD_FRAC):
    """Crop to the mask's padded bounding box."""
    mask = segment.require_mask()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(round((y1 - y0) * pad))
    pad_x = int(round((x1 - x0) * pad))

    return _apply_box(segment, x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y,
                      "crop_to_mask")


def _rotate(segment, degrees):
    """Rotate image and mask together about the frame center, expanding the canvas."""
    height, width = segment.shape
    matrix, size = rotation_matrix(degrees, width / 2.0, height / 2.0,
                                   segment.shape)

    image = apply_affine(np.asarray(segment.image), matrix, size,
                         flags=cv2.INTER_LINEAR)
    mask = None
    if segment.mask is not None:
        mask = apply_affine(segment.mask.astype(np.uint8), matrix, size) > 0

    info = {"degrees": degrees, "width": size[0], "height": size[1]}

    rotated = segment.replace(image=image,
                              mask=False if mask is None else mask,
                              applied=matrix)
    _visualize(segment, rotated, "rotate", info)
    return rotated, info


def _resize(segment, width=None, height=None, scale=None):
    """Resize image and mask together, preserving aspect ratio unless both dimensions are given."""
    source_height, source_width = segment.shape

    if scale is not None:
        new_width = max(1, int(round(source_width * scale)))
        new_height = max(1, int(round(source_height * scale)))
    else:
        if width is None:
            new_height = int(height)
            new_width = max(1, int(round(source_width * new_height / source_height)))
        elif height is None:
            new_width = int(width)
            new_height = max(1, int(round(source_height * new_width / source_width)))
        else:
            new_width, new_height = int(width), int(height)

    sx, sy = new_width / source_width, new_height / source_height
    matrix = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]])

    image = cv2.resize(np.asarray(segment.image), (new_width, new_height),
                       interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    mask = None
    if segment.mask is not None:
        mask = cv2.resize(segment.mask.astype(np.uint8), (new_width, new_height),
                          interpolation=cv2.INTER_NEAREST) > 0

    info = {"width": new_width, "height": new_height, "scale_x": sx, "scale_y": sy}

    resized = segment.replace(image=image,
                              mask=False if mask is None else mask,
                              applied=matrix)
    _visualize(segment, resized, "resize", info)
    return resized, info


def _visualize(segment, result, name, info):
    """Before and after frames, so a mis-specified region is obvious at a glance."""
    if segment.panel_sink is None:
        return

    before = np.asarray(segment.image).copy()
    after = np.asarray(result.image).copy()
    annotate(before, f"{name} in {info.get('source_width', segment.shape[1])}x"
                     f"{info.get('source_height', segment.shape[0])}")
    annotate(after, f"-> {after.shape[1]}x{after.shape[0]}")

    segment.emit_panel(side_by_side(before, after), name)
