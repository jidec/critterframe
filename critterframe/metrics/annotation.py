import logging
from functools import partial

import cv2
import numpy as np

from ..recipes import Metric
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
    return [" ".join(entries[index:index + _LEGEND_PER_LINE])
            for index in range(0, len(entries), _LEGEND_PER_LINE)]

# What click_two_points calls its two points unless told otherwise. The body
# axis is what a person is usually clicking in this package, so it's the
# default -- but it IS only a default, and the operation is named for what it
# asks (two points, in order) rather than for what any one project means by
# them.
DEFAULT_POINT_LABELS = ("head", "tail")


def _keys_for(labels):
    """`{key code: label}` for a label list, keys taken from LABEL_KEYS in order."""
    return {ord(key): label for key, label in zip(LABEL_KEYS, labels)}


def exclusive_label_annotation(labels, name=None, unit="category", requires_mask=True,
                               note=None, show_original=False):
    """
    Metric: show the segment and ask a human for exactly one of `labels`, by one
    keypress. The vocabulary is the caller's, e.g. whether a part's finished mask
    should reach an export:

        exclusive_label_annotation(
            ["good", "input_invalid", "wrong_region", "incomplete", "overflow"],
            name="abdomen_quality")

    The labels are part of the recipe: adding or renaming one later is a
    different recipe, and the labels given under the old list stop being
    current. Screening a finished segment, run it with the `from_part` and
    `transforms` the part was segmented with, so the panel is the frame the
    segmenter saw.

    - `labels` -- the distinct labels to choose between, at most 36. Keys are
      handed out in this order: 1-9, 0, then a-z. The legend is drawn on the
      panel.
    - `name` -- what to store the label under; `"exclusive_label_annotation"`
      by default.
    - `unit` -- recorded unit.
    - `requires_mask` -- True (default) for a label that describes the mask: it
      is asked only where a mask exists and stops being current when the mask
      is replaced. False for a label that describes the image, asked of every
      occurrence with an image and unaffected by resegmenting.
    - `note` -- optional text on how the labels are meant to be applied.
      Logged when a run starts and recorded with it, but not part of the
      recipe: rewording it leaves the labels already given current. What a
      label means belongs in `labels`.
    - `show_original` -- put the untouched image first in the panel, with the
      part outlined on it, where transforms have cropped or rotated the
      segment away from it. Display only: not part of the recipe and not
      recorded with the run, so a procedure that relies on it belongs in
      `note`.

    Returns the chosen label.
    """
    labels = [str(label) for label in labels]
    if not labels:
        raise ValueError("exclusive_label_annotation needs at least one label")
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"exclusive_label_annotation needs distinct labels, got {labels}")
    if len(labels) > len(LABEL_KEYS):
        raise ValueError(
            f"exclusive_label_annotation has {len(LABEL_KEYS)} keys to hand out, "
            f"got {len(labels)} labels")

    function = partial(
        _exclusive_label_of_a_mask if requires_mask else _exclusive_label_annotation,
        show_original=show_original)
    return Metric("exclusive_label_annotation", function,
                  {"labels": labels}, version="1", unit=unit, metric_name=name,
                  requires_mask=requires_mask, note=note)


def click_two_points(labels=DEFAULT_POINT_LABELS, name=None, unit="px_xy"):
    """
    Metric: have a human click two points on the segment, in order. Esc skips.

    Agnostic about what the points mean -- a body axis by default, but equally a
    wing chord or a scale bar. `labels` names them, and the names appear in the
    prompt and as the keys of the stored value.

    Both raw points are stored: a length and an angle are each recoverable from
    two points, but neither recovers the points. length_px and angle_deg ride
    along as a convenience, since validation.metrics compares numbers.

    - `labels` -- the two point names, in click order.

    Returns {<first>, <second>, "length_px", "angle_deg"}, all None if skipped.
    Points are [x, y] in the current frame, normally original image coordinates,
    whatever size the window was fitted to (see `panels.DISPLAY_MAX`).
    angle_deg is measured in image coordinates (y DOWN), so +90 means the second
    point is directly below the first.

    The unit is "px_xy", which is not convertible: length_px stays in pixels in a
    units="mm" export while body_length converts. Right for a label whose job is
    grading a pipeline measured in pixels.
    """
    labels = [str(label) for label in labels]
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError(
            f"click_two_points needs two distinct labels, got {labels}"
        )

    return Metric("click_two_points", _click_two_points, {"labels": labels},
                  version="1", unit=unit, metric_name=name)


def _panel(segment):
    """
    Image, mask, and overlay side by side -- the standard annotation view.

    Falls back to the image alone when there's no mask yet: a label that
    describes the image (exclusive_label_annotation(requires_mask=False)) can
    be asked before segmentation, so a screening panel has to work without
    one; click_two_points still requires a mask (it clicks points ON the
    segment) and never reaches this fallback.
    """
    image = np.asarray(segment.image)
    if segment.mask is None:
        return image
    mask = segment.mask
    return side_by_side(image, mask_to_bgr(mask), overlay_mask(image, mask))


def _display_panel(segment, show_original=False):
    """
    The standard panel at the size the window shows it. With `show_original`,
    the untouched image comes first (see `_original_view`).
    """
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
    working = cv2.resize(working, (max(1, int(width * scale)), shown_height),
                         interpolation=interpolation)
    return side_by_side(_original_view(segment, shown_height), working)


# The outline drawn on the original image: yellow reads on dark and light alike.
OUTLINE_COLOR = (0, 255, 255)


def _has_moved(segment):
    """
    Whether a transform has moved the segment away from the image it started
    from. False where no original was kept: the working image then IS the original.
    """
    original = segment.original_image
    if original is None:
        return False
    return (not np.allclose(segment.matrix, np.eye(2, 3))
            or np.asarray(original).shape[:2] != np.asarray(segment.image).shape[:2])


def _original_view(segment, height):
    """
    The image the segment started from, with the part outlined on it, scaled to
    `height`; None where there is nothing to add (see `_has_moved`).
    """
    if not _has_moved(segment):
        return None
    original = np.asarray(segment.original_image)

    view = (cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2
            else original.copy())
    outline = segment.mask_in_original_coordinates().astype(np.uint8)
    contours, _hierarchy = cv2.findContours(outline, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    # Thick enough to survive the resize below, whatever the photo's size.
    thickness = max(1, round(view.shape[0] / max(1, height))) + 1
    cv2.drawContours(view, contours, -1, OUTLINE_COLOR, thickness)

    # Rounded down, so the joined panel never exceeds the box it was sized for.
    width = max(1, int(view.shape[1] * height / view.shape[0]))
    interpolation = cv2.INTER_AREA if height < view.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(view, (width, height), interpolation=interpolation)


def _wait_for_key(valid_keys):
    """
    Block (no timeout) until one of valid_keys is pressed, ignoring anything else.

    DELIBERATELY duplicated in segmentation.manual rather than shared: the tests
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


def _ask(segment, panel, prompt, keys):
    """Show a panel, wait for one of `keys`, return the label it maps to."""
    window = f"{segment.occurrence_id} {segment.part} - {prompt}"
    cv2.imshow(window, panel)
    key = _wait_for_key(set(keys))
    cv2.destroyWindow(window)
    return keys[key]


def _ask_flag(segment, keys, prompt, show_original=False):
    """Show the standard panel with `keys`' legend on it, return the chosen label."""
    # Legend drawn after fitting, so it stays readable whatever the image size.
    panel = _display_panel(segment, show_original=show_original).copy()
    for line, text in enumerate(_legend_lines(keys)):
        annotate(panel, text, line=line)
    return _ask(segment, panel, prompt, keys)


def _exclusive_label_annotation(segment, labels, show_original=False):
    return _ask_flag(segment, _keys_for(labels), "label? (legend on image)",
                     show_original=show_original)


def _exclusive_label_of_a_mask(segment, labels, show_original=False):
    if segment.mask is None:
        raise ValueError(f"{segment.occurrence_id} has no mask yet -- this label "
                         "describes one (requires_mask=True)")
    return _exclusive_label_annotation(segment, labels, show_original=show_original)


def _click_two_points(segment, labels=DEFAULT_POINT_LABELS):
    image, scale = fit_for_display(
        overlay_mask(np.asarray(segment.image), segment.require_mask()))
    height, width = segment.require_mask().shape
    window = (f"{segment.occurrence_id} {segment.part} - click {labels[0]}, "
              f"then {labels[1]} (Esc=skip)")
    cv2.imshow(window, image)

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            # Stored in the segment's own pixels, not the fitted window's.
            points.append((min(width - 1, max(0, round(x / scale))),
                           min(height - 1, max(0, round(y / scale)))))
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
        cv2.waitKey(300)   # let the second marker stay visible briefly
    cv2.destroyWindow(window)

    return _skipped_pair(labels) if skipped else _point_pair(labels, points)


def _point_pair(labels, points):
    """
    The value two clicked points produce: both positions, the distance between
    them, and the angle of the line they define.

    Separate from the clicking so the arithmetic is reachable without a window.
    That matters more here than it looks: these are the only computed numbers
    this module produces, and the angle convention is the kind of thing that is
    wrong for months before anyone notices. It follows image coordinates, where
    y increases DOWNWARD, so a second point below the first is +90 degrees, not
    -90.

    - `labels` -- the two names the positions are stored under.
    - `points` -- [(x0, y0), (x1, y1)] in the segment's current coordinates.
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
    """
    The same shape with nothing in it, for an occurrence the annotator skipped.

    Nulls rather than no row at all: "this one was looked at and passed over" is
    a different fact from "this one was never reached", and only the first is
    recoverable from a value.
    """
    return {labels[0]: None, labels[1]: None,
            "length_px": None, "angle_deg": None}
