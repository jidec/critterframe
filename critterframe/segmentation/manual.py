"""
Manual segmentation: a human draws or corrects the mask.

Drawing a mask by hand is an ALTERNATIVE SEGMENTATION, not a separate system.
draw_mask() and correct_mask() are Segmentation operations exactly like
segment(groundedsam2()) is -- they take a segment and give it a mask, they
compose with transforms the same way, and they record a run and a recipe hash
the same way. What differs is only which one a recipe names, and therefore what
hash the resulting mask carries.

That matters for validation. A reference mask built by a human and a canonical
mask built by a model are two masks of the same occurrence-part produced by two
recipes; validation just compares them (see critterframe.validation.masks). No
special "reference mode" runs through the pipeline, because there doesn't
need to be one.

These operations open an OpenCV window and block on a person, so they belong in
a run a human is sitting through -- typically over a subset of a few dozen
hand-picked occurrences, not the whole project.
"""

import logging

import cv2
import numpy as np

from ..recipes import Segmentation
from ..visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)

DEFAULT_BRUSH_RADIUS = 8


def correct_mask(brush_radius=DEFAULT_BRUSH_RADIUS):
    """
    Operation: show the current mask over the image and let a human fix it.

    Left-drag erases pixels that were wrongly included (an extra leg, a shadow,
    a second organism in frame); right-drag paints in pixels that were missed
    (a dropped wing, an under-segmented edge). 's' saves, Esc cancels and keeps
    the mask unchanged.

    Covers both of a segmenter's failure directions rather than only trimming
    what it over-included, so the result is usable as a reference either way,
    not just a trimmed-down copy of what the model over-included.

    The mask being corrected is whatever the segment arrives with, which in a
    run means run_segments(from_part=...) -- that is what loads the stored mask
    to start from. Given no from_part a segment starts empty and this behaves
    like draw_mask(), which is a silent difference worth knowing about: the
    window looks the same either way.

    Point this at crops that HAVE a definable correction. Two organisms in
    frame, no organism, or one running off the frame edge: there is no single
    complete boundary to paint, so whatever gets painted is invented, and it
    lands in the reference table and drags down the IoU validate_masks reports
    as if the segmenter had got something wrong. Screen first
    (metrics.annotation.annotate_flags), then run this over the occurrences
    flagged "usable" -- selectionhelpers.occurrences_matching turns the one into
    the other.

    The correction IS the mask grade, which is why there's no metric asking a
    person to rate one: paint the boundary and validation.masks reports the IoU
    against it.

    brush_radius -- brush size in pixels.
    """
    return Segmentation("correct_mask", _paint,
                        {"brush_radius": brush_radius, "start_empty": False},
                        version="1")


def draw_mask(brush_radius=DEFAULT_BRUSH_RADIUS):
    """
    Operation: have a human paint a mask from scratch, with no model involved.

    For organisms no available model segments acceptably, and for building a
    first training set where there's nothing to correct yet. Same window and
    controls as correct_mask(), just starting from an empty mask.

    brush_radius -- brush size in pixels.
    """
    return Segmentation("draw_mask", _paint,
                        {"brush_radius": brush_radius, "start_empty": True},
                        version="1")


def _wait_for_key(valid_keys):
    """Block (no timeout) until one of valid_keys is pressed, ignoring anything else."""
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in valid_keys:
            return key


def _paint(segment, brush_radius=DEFAULT_BRUSH_RADIUS, start_empty=False):
    """
    The interactive painting loop behind both correct_mask() and draw_mask().

    Returns (segment, info) with info carrying area before/after,
    removed_fraction, added_fraction, and iou against the starting mask.
    Removed and added are reported separately rather than as one net figure
    because they answer different questions about the segmenter being graded --
    a mask that was 20% too big and one that was 20% too small are not the same
    failure.
    """
    image = np.asarray(segment.image)
    if start_empty or segment.mask is None:
        original = np.zeros(image.shape[:2], dtype=bool)
    else:
        original = segment.mask

    edited = (original.astype(np.uint8) * 255).copy()
    painting = {"mode": None}       # "erase", "add", or None when not dragging
    instructions = "left=erase right=add ('s'=save, Esc=cancel)"
    window = f"{segment.occurrence_id} {segment.part} - {instructions}"

    def redraw():
        cv2.imshow(window, overlay_mask(image, edited))

    def paint_at(x, y, value):
        cv2.circle(edited, (x, y), brush_radius, value, -1)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            painting["mode"] = "erase"
            paint_at(x, y, 0)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            painting["mode"] = "add"
            paint_at(x, y, 255)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and painting["mode"] is not None:
            paint_at(x, y, 0 if painting["mode"] == "erase" else 255)
            redraw()
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            painting["mode"] = None

    cv2.imshow(window, image)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    key = _wait_for_key({ord("s"), 27})
    cv2.destroyWindow(window)

    corrected = original if key == 27 else (edited > 0)
    if not corrected.any():
        raise ValueError("no mask was drawn")

    area_before = int(original.sum())
    area_after = int(corrected.sum())
    removed = int((original & ~corrected).sum())
    added = int((corrected & ~original).sum())
    intersection = int((original & corrected).sum())
    union = int((original | corrected).sum())

    info = {
        "cancelled": key == 27,
        "area_before": area_before,
        "area_after": area_after,
        "removed_fraction": (removed / area_before) if area_before else 0.0,
        "added_fraction": (added / area_before) if area_before else 0.0,
        "iou": (intersection / union) if union else 1.0,
    }

    _visualize(segment, original, corrected, info)
    return segment.replace(mask=corrected), info


def _visualize(segment, original, corrected, info):
    """Kept pixels white, erased red, added green."""
    if segment.panel_sink is None:
        return

    panel = np.zeros((*original.shape, 3), dtype=np.uint8)
    panel[original & corrected] = (255, 255, 255)
    panel[original & ~corrected] = (0, 0, 255)
    panel[~original & corrected] = (0, 255, 0)
    annotate(panel, f"iou {info['iou']:.2f} "
                    f"-{info['removed_fraction']:.1%} +{info['added_fraction']:.1%}")

    segment.emit_panel(panel, "manual")
