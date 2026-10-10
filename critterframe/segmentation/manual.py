"""Draw or correct a mask by hand: draw_mask and correct_mask."""

import logging

import cv2
import numpy as np

from ..maskops import mask_iou
from ..core.recipes import Segmentation
from ..visualization.panels import (
    ADDED_COLOR,
    REMOVED_COLOR,
    annotate,
    diff_panel,
    fit_for_display,
    overlay_mask,
)

logger = logging.getLogger(__name__)

DEFAULT_BRUSH_RADIUS = 8


def correct_mask(brush_radius=DEFAULT_BRUSH_RADIUS):
    """Operation: show the mask over the image and let a person fix it.

    Left-drag erases, right-drag paints, `+` and `-` resize the brush, `s` saves and Esc
    cancels. The mask corrected is the one the segment arrives with, so use it with
    `run_segments(from_part=...)`; with none, it starts empty like `draw_mask`.

    Args:
        brush_radius: Starting brush size in screen pixels.
    """
    return Segmentation(
        "correct_mask",
        _paint,
        {"brush_radius": brush_radius, "start_empty": False},
        version="1",
        deterministic=False,
    )


def draw_mask(brush_radius=DEFAULT_BRUSH_RADIUS):
    """Operation: have a person paint a mask from scratch.

    Same window and controls as `correct_mask`.

    Args:
        brush_radius: Starting brush size in screen pixels.
    """
    return Segmentation(
        "draw_mask",
        _paint,
        {"brush_radius": brush_radius, "start_empty": True},
        version="1",
        deterministic=False,
    )


def _wait_for_key(valid_keys):
    """Block until one of the valid keys is pressed.

    Duplicated in `metrics.annotation` on purpose: tests replace `cv2` in this module's
    namespace, and a shared copy elsewhere would keep the real one and hang.
    """
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in valid_keys:
            return key


def _paint(segment, brush_radius=DEFAULT_BRUSH_RADIUS, start_empty=False):
    """Run the painting loop behind `correct_mask` and `draw_mask`.

    Returns:
        `(segment, info)`, with the area before and after, `removed_fraction`,
        `added_fraction`, and `iou` against the starting mask.
    """
    image = np.asarray(segment.image)
    if start_empty or segment.mask is None:
        original = np.zeros(image.shape[:2], dtype=bool)
    else:
        original = segment.mask

    edited = (original.astype(np.uint8) * 255).copy()
    painting = {"mode": None, "last": None}  # mode: "erase", "add", or None
    brush = {"radius": brush_radius}
    instructions = "left=erase right=add (+/-=brush, 's'=save, Esc=cancel)"
    window = f"{segment.occurrence_id} {segment.part} - {instructions}"
    # The window shows a resized copy; `edited` stays at the segment's own
    # resolution, so every mouse position is mapped back before painting.
    _shown, scale = fit_for_display(image)
    height, width = image.shape[:2]

    def redraw():
        shown, _scale = fit_for_display(overlay_mask(image, edited))
        cv2.imshow(window, annotate(shown, f"brush radius: {brush['radius']}"))

    def to_image(x, y):
        return (min(width - 1, max(0, round(x / scale))), min(height - 1, max(0, round(y / scale))))

    def paint_at(x, y, value):
        point = to_image(x, y)
        radius = max(1, round(brush["radius"] / scale))
        if painting["last"] is not None:
            # Mouse-move events are sparse; join them so a fast drag leaves no gaps.
            cv2.line(edited, painting["last"], point, value, 2 * radius + 1)
        cv2.circle(edited, point, radius, value, -1)
        painting["last"] = point

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            painting["mode"], painting["last"] = "erase", None
            paint_at(x, y, 0)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            painting["mode"], painting["last"] = "add", None
            paint_at(x, y, 255)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and painting["mode"] is not None:
            paint_at(x, y, 0 if painting["mode"] == "erase" else 255)
            redraw()
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            painting["mode"], painting["last"] = None, None

    cv2.imshow(window, _shown)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    # '=' and '-' need no shift key, so both they and their shifted partners
    # ('+' and '_') grow/shrink the brush; growing and shrinking loop back to
    # wait for another key instead of ending the session.
    grow_keys = {ord("+"), ord("=")}
    shrink_keys = {ord("-"), ord("_")}
    while True:
        key = _wait_for_key({ord("s"), 27} | grow_keys | shrink_keys)
        if key in grow_keys:
            brush["radius"] += 1
        elif key in shrink_keys:
            brush["radius"] = max(1, brush["radius"] - 1)
        else:
            break
        redraw()
    cv2.destroyWindow(window)

    corrected = original if key == 27 else (edited > 0)
    if not corrected.any():
        raise ValueError("no mask was drawn")

    area_before = int(original.sum())
    area_after = int(corrected.sum())
    removed = int((original & ~corrected).sum())
    added = int((corrected & ~original).sum())

    info = {
        "cancelled": key == 27,
        "area_before": area_before,
        "area_after": area_after,
        "removed_fraction": (removed / area_before) if area_before else 0.0,
        "added_fraction": (added / area_before) if area_before else 0.0,
        "iou": mask_iou(original, corrected),
    }

    _visualize(segment, original, corrected, info)
    return segment.replace(mask=corrected), info


def _visualize(segment, original, corrected, info):
    """Emit a panel: kept pixels white, erased red, added green."""
    if segment.panel_sink is None:
        return

    panel = diff_panel(original, corrected, only_mask=REMOVED_COLOR, only_other=ADDED_COLOR)
    annotate(panel, f"iou {info['iou']:.2f} -{info['removed_fraction']:.1%} +{info['added_fraction']:.1%}")

    segment.emit_panel(panel, "manual")
