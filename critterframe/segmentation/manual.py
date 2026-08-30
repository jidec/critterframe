"""
Draw/correct a mask by hand -- an alternative segmentation, not a separate
system.

draw_mask() and correct_mask() are Segmentation operations exactly as
segment(groundedsam2()) is: they compose with transforms the same way and
record a run and a recipe hash the same way. That is what lets validation
simply compare a human's mask against a model's.

Both open an OpenCV window and block on a person, so scope them to a subset of
a few dozen occurrences rather than a whole project.
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

    Left-drag erases pixels wrongly included, right-drag paints in pixels that
    were missed; 's' saves, Esc cancels. Covering both failure directions is
    what makes the result usable as a reference either way.

    The mask corrected is whatever the segment arrives with, which in a run means
    run_segments(from_part=...). With no from_part the segment starts empty and
    this behaves like draw_mask() -- a silent difference, since the window looks
    the same.

    Point this at crops that HAVE a definable correction. With two organisms, no
    organism, or one running off the edge there is no single boundary to paint,
    so whatever gets painted is invented and then drags down the IoU
    validate_masks reports as if the segmenter had erred. Screen with
    annotate_flags first, then run this over the crops flagged usable.

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
    """
    Block (no timeout) until one of valid_keys is pressed, ignoring anything else.

    DELIBERATELY duplicated in metrics.annotation rather than shared: the tests
    for both interactive operations stub the GUI with
    monkeypatch.setattr(<this module>, "cv2", FakeCv2(...)), which rebinds `cv2`
    in THIS module's namespace only. Moved into visualization.panels, the shared
    copy would keep its own reference to the real cv2, the fake would never
    reach it, and every interactive test would block on a real waitKey until
    pytest-timeout killed the run. Four lines is cheaper than that.
    """
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
