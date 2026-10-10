"""Human labels as metrics: exclusive_label_annotation, click_two_points."""

import logging
from functools import partial

import cv2
import numpy as np

from ..core.recipes import Metric
from ..visualization.panels import (
    DISPLAY_MAX,
    annotate,
    fit_for_display,
    mask_to_bgr,
    overlay_mask,
    side_by_side,
)

logger = logging.getLogger(__name__)

# The keys exclusive_label_annotation hands out, in order: one per label, so
# this is also how many labels one vocabulary can hold.
LABEL_KEYS = "1234567890abcdefghijklmnopqrstuvwxyz"

# A key map's legend, wrapped onto a few short lines rather than one long
# prompt -- a dozen entries is too many for a window title to stay readable,
# so it's drawn on the panel itself instead (see _ask_flag).
_LEGEND_PER_LINE = 4


def _legend_lines(keys):
    entries = [f"{chr(key)}={name}" for key, name in keys.items()]
    return [
        " ".join(entries[index : index + _LEGEND_PER_LINE])
        for index in range(0, len(entries), _LEGEND_PER_LINE)
    ]


# What click_two_points calls its two points unless told otherwise. The body
# axis is what a person is usually clicking in this package, so it's the
# default -- but it IS only a default, and the operation is named for what it
# asks (two points, in order) rather than for what any one project means by
# them.
DEFAULT_POINT_LABELS = ("head", "tail")


def _keys_for(labels):
    """Return `{key code: label}` for a label list, keys taken from `LABEL_KEYS` in order."""
    return {ord(key): label for key, label in zip(LABEL_KEYS, labels)}


def exclusive_label_annotation(
    labels, name=None, unit="category", requires_mask=True, note=None, show_original=False
):
    """Metric: show the segment and ask a person for exactly one of `labels`, by one keypress.

    The labels are part of the recipe, so adding or renaming one starts a new set of labels.

    Args:
        labels: The distinct labels to choose between, at most 36. Keys are handed out in
            order: 1-9, 0, then a-z.
        name: Name to store the label under; `"exclusive_label_annotation"` if None.
        unit: Recorded unit.
        requires_mask: True for a label that describes the mask: asked only where one
            exists, and no longer current once it is replaced. False for a label that
            describes the image, asked of every occurrence with an image.
        note: How the labels are meant to be applied. Recorded with the run, not hashed.
        show_original: Put the untouched image first in the panel, with the part outlined,
            where transforms have moved the segment. Display only.
    """
    labels = [str(label) for label in labels]
    if not labels:
        raise ValueError("exclusive_label_annotation needs at least one label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"exclusive_label_annotation needs distinct labels, got {labels}")
    if len(labels) > len(LABEL_KEYS):
        raise ValueError(
            f"exclusive_label_annotation has {len(LABEL_KEYS)} keys to hand out, got {len(labels)} labels"
        )

    function = partial(
        _exclusive_label_of_a_mask if requires_mask else _exclusive_label_annotation,
        show_original=show_original,
    )
    return Metric(
        "exclusive_label_annotation",
        function,
        {"labels": labels},
        version="1",
        unit=unit,
        metric_name=name,
        requires_mask=requires_mask,
        note=note,
    )


def click_two_points(labels=DEFAULT_POINT_LABELS, name=None, unit="px_xy"):
    """Metric: have a person click two points on the segment, in order. Esc skips.

    Args:
        labels: The two point names, in click order.
        name: Name to store the value under.
        unit: Recorded unit. The default `"px_xy"` is not converted by a `units="mm"` export.

    Returns:
        The metric. Its value is `{<first>, <second>, "length_px", "angle_deg"}`, all None
        if skipped. Points are `[x, y]` in the segment's frame; `angle_deg` follows image
        coordinates, so +90 means the second point is directly below the first.
    """
    labels = [str(label) for label in labels]
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError(f"click_two_points needs two distinct labels, got {labels}")

    return Metric(
        "click_two_points", _click_two_points, {"labels": labels}, version="1", unit=unit, metric_name=name
    )


def _panel(segment):
    """Return the image, mask and overlay side by side, or the image alone when there is no mask."""
    image = np.asarray(segment.image)
    if segment.mask is None:
        return image
    mask = segment.mask
    return side_by_side(image, mask_to_bgr(mask), overlay_mask(image, mask))


def _display_panel(segment, show_original=False):
    """Return the standard panel at window size, with the untouched image first if `show_original`."""
    working = _panel(segment)
    if not (show_original and segment.mask is not None and _has_moved(segment)):
        return fit_for_display(working)[0]

    # Each half is resized once, from its own resolution, to the height the two
    # share on screen. Joining them at the crop's height first would shrink the
    # photo to a thumbnail and then enlarge that.
    height, width = working.shape[:2]
    original_height, original_width = np.asarray(segment.original_image).shape[:2]
    joined_width = width + original_width * height / original_height
    scale = min(DISPLAY_MAX[0] / joined_width, DISPLAY_MAX[1] / height)
    shown_height = max(1, int(height * scale))
    interpolation = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
    working = cv2.resize(working, (max(1, int(width * scale)), shown_height), interpolation=interpolation)
    return side_by_side(_original_view(segment, shown_height), working)


# The outline drawn on the original image: yellow reads on dark and light alike.
OUTLINE_COLOR = (0, 255, 255)


def _has_moved(segment):
    """Return whether a transform has moved the segment away from the image it started from."""
    original = segment.original_image
    if original is None:
        return False
    return (
        not np.allclose(segment.matrix, np.eye(2, 3))
        or np.asarray(original).shape[:2] != np.asarray(segment.image).shape[:2]
    )


def _original_view(segment, height):
    """Return the original image with the part outlined, scaled to `height`, or None if nothing moved."""
    if not _has_moved(segment):
        return None
    original = np.asarray(segment.original_image)

    view = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
    outline = segment.mask_in_original_coordinates().astype(np.uint8)
    contours, _hierarchy = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Thick enough to survive the resize below, whatever the photo's size.
    thickness = max(1, round(view.shape[0] / max(1, height))) + 1
    cv2.drawContours(view, contours, -1, OUTLINE_COLOR, thickness)

    # Rounded down, so the joined panel never exceeds the box it was sized for.
    width = max(1, int(view.shape[1] * height / view.shape[0]))
    interpolation = cv2.INTER_AREA if height < view.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(view, (width, height), interpolation=interpolation)


def _wait_for_key(valid_keys):
    """Block until one of the valid keys is pressed.

    Duplicated in `segmentation.manual` on purpose: tests replace `cv2` in this module's
    namespace, and a shared copy elsewhere would keep the real one and hang.
    """
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in valid_keys:
            return key


def _ask(segment, panel, prompt, keys):
    """Show a panel, wait for one of `keys`, and return the label it maps to."""
    window = f"{segment.occurrence_id} {segment.part} - {prompt}"
    cv2.imshow(window, panel)
    key = _wait_for_key(set(keys))
    cv2.destroyWindow(window)
    return keys[key]


def _ask_flag(segment, keys, prompt, show_original=False):
    """Show the standard panel with the keys' legend on it, and return the chosen label."""
    # Legend drawn after fitting, so it stays readable whatever the image size.
    panel = _display_panel(segment, show_original=show_original).copy()
    for line, text in enumerate(_legend_lines(keys)):
        annotate(panel, text, line=line)
    return _ask(segment, panel, prompt, keys)


def _exclusive_label_annotation(segment, labels, show_original=False):
    return _ask_flag(segment, _keys_for(labels), "label? (legend on image)", show_original=show_original)


def _exclusive_label_of_a_mask(segment, labels, show_original=False):
    if segment.mask is None:
        raise ValueError(
            f"{segment.occurrence_id} has no mask yet -- this label describes one (requires_mask=True)"
        )
    return _exclusive_label_annotation(segment, labels, show_original=show_original)


def _click_two_points(segment, labels=DEFAULT_POINT_LABELS):
    image, scale = fit_for_display(overlay_mask(np.asarray(segment.image), segment.require_mask()))
    height, width = segment.require_mask().shape
    window = f"{segment.occurrence_id} {segment.part} - click {labels[0]}, then {labels[1]} (Esc=skip)"
    cv2.imshow(window, image)

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            # Stored in the segment's own pixels, not the fitted window's.
            points.append(
                (min(width - 1, max(0, round(x / scale))), min(height - 1, max(0, round(y / scale))))
            )
            color = (0, 255, 0) if len(points) == 1 else (0, 0, 255)
            cv2.circle(image, (x, y), 4, color, -1)
            cv2.imshow(window, image)

    cv2.setMouseCallback(window, on_click)

    skipped = False
    while len(points) < 2:
        if cv2.waitKey(20) & 0xFF == 27:
            skipped = True
            break
    if not skipped:
        cv2.waitKey(300)  # let the second marker stay visible briefly
    cv2.destroyWindow(window)

    return _skipped_pair(labels) if skipped else _point_pair(labels, points)


def _point_pair(labels, points):
    """Return the value two clicked points produce: both positions, their distance and angle.

    The angle follows image coordinates, where y increases downward: a second point below
    the first is +90 degrees.
    """
    (x0, y0), (x1, y1) = points
    dx, dy = x1 - x0, y1 - y0
    return {
        labels[0]: [x0, y0],
        labels[1]: [x1, y1],
        "length_px": float(np.hypot(dx, dy)),
        "angle_deg": float(np.degrees(np.arctan2(dy, dx))),
    }


def _skipped_pair(labels):
    """Return the same shape with None values, for an occurrence the annotator skipped."""
    return {labels[0]: None, labels[1]: None, "length_px": None, "angle_deg": None}
