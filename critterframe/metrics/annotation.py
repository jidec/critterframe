import logging

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import annotate, mask_to_bgr, overlay_mask, side_by_side

logger = logging.getLogger(__name__)

# Why a crop shouldn't be trusted, or that it should. Anything other than
# "usable" means downstream traits from this occurrence are suspect, but the
# REASONS are worth telling apart -- a metric that catches every cut-off
# organism while missing every non-organism is a different (and more useful)
# instrument than one aggregate "bad" rate would suggest.
#
# Two different kinds of "not usable" live in this one list. not_an_organism,
# cut_off, multiple_organisms, broken_body, and obscured mean there is no
# single complete boundary to draw at all. wrong_life_stage, bad_angle, dead,
# blurry, overexposed, underexposed, and wrong_organism_for_project usually DO
# still segment into one clean boundary -- what disqualifies them is the
# specimen or the image, not the geometry. Both kinds are treated the same
# way today (excluded from reference masks, training, and the export
# alike -- see validation.filters.BAD_FLAGS) rather than split, so a
# segmentable-but-invalid crop currently costs a training example it didn't
# strictly have to. Worth revisiting if that cost turns out to matter more
# than the simplicity of one flat "usable or not" gate.
FLAG_KEYS = {
    ord("1"): "usable",
    ord("2"): "not_an_organism",
    ord("3"): "cut_off",
    ord("4"): "multiple_organisms",
    ord("5"): "wrong_life_stage",
    ord("6"): "bad_angle",
    ord("7"): "dead",
    ord("8"): "broken_body",
    ord("9"): "obscured",
    ord("0"): "blurry",
    ord("a"): "overexposed",
    ord("b"): "underexposed",
    ord("c"): "wrong_organism_for_project",
}

# FLAG_KEYS's legend, wrapped onto a few short lines rather than one long
# prompt -- 13 entries is too many for a window title to stay readable, so
# it's drawn on the panel itself instead (see _usability_annotation).
_LEGEND_PER_LINE = 4


def _legend_lines():
    entries = [f"{chr(key)}={name}" for key, name in FLAG_KEYS.items()]
    return [" ".join(entries[index:index + _LEGEND_PER_LINE])
            for index in range(0, len(entries), _LEGEND_PER_LINE)]

# What click_two_points calls its two points unless told otherwise. The body
# axis is what a person is usually clicking in this package, so it's the
# default -- but it IS only a default, and the operation is named for what it
# asks (two points, in order) rather than for what any one project means by
# them.
DEFAULT_POINT_LABELS = ("head", "tail")


def usability_annotation(name=None, unit="category"):
    """
    Metric: show the image, mask, and overlay side by side and ask a human to
    classify the occurrence. The on-screen legend is generated from FLAG_KEYS,
    so this list and the prompt can never drift apart:

      1 = usable                      a single, complete organism
      2 = not_an_organism             nothing that should have been ingested
      3 = cut_off                     an organism, but running off the frame edge
      4 = multiple_organisms          more than one in frame
      5 = wrong_life_stage            an organism, but not the stage this project studies
      6 = bad_angle                   photographed from an angle that can't be measured reliably
      7 = dead                        a dead specimen
      8 = broken_body                 missing or damaged body parts
      9 = obscured                    one organism, partly hidden behind debris/vegetation/another organism
      0 = blurry                      too out of focus to trust
      a = overexposed                 too bright to trust
      b = underexposed                too dark to trust
      c = wrong_organism_for_project  something's there and segments fine, but isn't this project's subject

    A crop matching more than one reason gets whichever one also explains why
    the segmentation itself can't be trusted -- a crop that's both cut off and
    blurry is flagged cut_off, not blurry.

    This is the SCREENING pass, and it comes before any other human work --
    including segmentation itself: requires_mask=False, so run_metrics
    measures every occurrence with an image, whether or not it has been
    segmented yet. Judging a crop's usability doesn't need a boundary, and
    most of what this flags (dead, wrong_life_stage, blurry, ...) is exactly
    what you'd want to catch BEFORE spending a segmentation model's time on
    it, not after. The panel shows image+mask+overlay when a mask exists and
    the image alone when it doesn't.

    A flag other than "usable" excludes an occurrence from reference masks,
    training, and the export alike (see validation.filters.BAD_FLAGS) --
    whether because there's no single complete boundary to draw (not_an_organism,
    cut_off, multiple_organisms, broken_body, obscured), or because a valid
    boundary exists but the specimen or the image itself makes it worthless to
    measure (wrong_life_stage, bad_angle, dead, blurry, overexposed,
    underexposed, wrong_organism_for_project). Screen the whole sample, then
    point the reference passes at the usable ones -- an unscreened sample has
    nothing for validation.filters to calibrate a QC cutoff against.
    """
    return Metric("usability_annotation", _usability_annotation, version="1",
                  unit=unit, metric_name=name, requires_mask=False)


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
    Points are [x, y] in the current frame, normally original image coordinates.
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

    Falls back to the image alone when there's no mask yet: usability_annotation
    runs before segmentation (requires_mask=False), so a screening panel has to
    work without one; click_two_points still requires a mask (it clicks points
    ON the segment) and never reaches this fallback.
    """
    image = np.asarray(segment.image)
    if segment.mask is None:
        return image
    mask = segment.mask
    return side_by_side(image, mask_to_bgr(mask), overlay_mask(image, mask))


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


def _usability_annotation(segment):
    panel = _panel(segment)
    for line, text in enumerate(_legend_lines()):
        annotate(panel, text, line=line)
    return _ask(segment, panel, "usability flag? (legend on image)", FLAG_KEYS)

def _click_two_points(segment, labels=DEFAULT_POINT_LABELS):
    image = overlay_mask(np.asarray(segment.image), segment.require_mask())
    window = (f"{segment.occurrence_id} {segment.part} - click {labels[0]}, "
              f"then {labels[1]} (Esc=skip)")
    cv2.imshow(window, image)

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))
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
