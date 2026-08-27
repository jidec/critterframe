"""
Human annotation metrics: a person looks at the segment and answers.

No model, no heuristic. These exist to produce the labels everything else is
graded against -- whether a crop should have been filtered, where the body axis
really runs.

There is deliberately no "grade this mask perfect/good/bad" metric. A coarse
category asks a person to compress a judgement they can't make consistently
across a session, and it answers a question already answered better elsewhere:
correct the mask instead (segmentation.manual.correct_mask) and validation.masks
reports the IoU, which is the same judgement measured rather than estimated,
per-occurrence, and comparable between two versions of a segmenter.

They're metrics like any other, which is the point: a human label is a derived
value associated with an occurrence and a part, so it stores in the same table
as body length, exports in the same CSV, and can be compared against an
automated metric without any special plumbing (see validation.filters, which
calibrates automated QC thresholds against exactly these labels).

Each opens an OpenCV window and blocks on a keypress or click, so a run using
them is paced by a person -- scope it to a subset of a few dozen occurrences
rather than a whole project. run_metrics writes each occurrence's answers as
they're given, so stopping partway keeps everything answered so far, and
resuming skips it.
"""

import logging

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import mask_to_bgr, overlay_mask, side_by_side

logger = logging.getLogger(__name__)

# Why a crop shouldn't be trusted, or that it should. Anything other than
# "usable" means downstream traits from this occurrence are suspect, but the
# REASONS are worth telling apart -- a metric that catches every cut-off
# organism while missing every non-organism is a different (and more useful)
# instrument than one aggregate "bad" rate would suggest.
FLAG_KEYS = {
    ord("1"): "usable",
    ord("2"): "not_an_organism",
    ord("3"): "cut_off",
    ord("4"): "multiple_organisms",
}

# What click_two_points calls its two points unless told otherwise. The body
# axis is what a person is usually clicking in this package, so it's the
# default -- but it IS only a default, and the operation is named for what it
# asks (two points, in order) rather than for what any one project means by
# them.
DEFAULT_POINT_LABELS = ("head", "tail")


def annotate_flags(name=None, unit="category"):
    """
    Metric: show the image, mask, and overlay side by side and ask a human to
    classify the occurrence.

      1 = usable              a single, complete organism
      2 = not_an_organism     nothing that should have been ingested
      3 = cut_off             an organism, but running off the frame edge
      4 = multiple_organisms  more than one in frame

    If a crop is BOTH cut off and has several organisms, flag it cut_off -- of
    the two, that's the one that also explains why the segmentation can't be
    trusted.

    These labels are the reference automated QC metrics are calibrated
    against; see metrics.quality and validation.filters.

    This is the SCREENING pass, and it comes before any other human work on a
    sample, because it says which crops the expensive human questions can even
    be asked of. A flag other than "usable" means there is no single complete
    organism in frame, so there is no boundary to correct and no reference mask
    to make either (segmentation.manual.correct_mask) -- and no body axis to
    click. Screen the whole sample, then point the reference passes at the
    usable ones (selectionhelpers.occurrences_matching).

    Screening the WHOLE sample matters as much as the order does: the flagged
    crops are the positives validation.filters scores a QC cutoff against, so a
    sample they were left out of has nothing to detect and can't calibrate
    anything.
    """
    return Metric("annotate_flags", _annotate_flags, version="1", unit=unit,
                  metric_name=name)


def click_two_points(labels=DEFAULT_POINT_LABELS, name=None, unit="px_xy"):
    """
    Metric: have a human click two points on the segment, in order. Esc skips,
    for an occurrence where one of them isn't visible.

    Agnostic about what the two points MEAN -- the head and tail ends of a body
    axis by default, but equally the two ends of a wing chord, the tips of a
    scale bar, or the gap between two structures. `labels` names them, and the
    names appear in the window prompt and as the keys of the stored value, so
    one operation covers every "click these two spots" question instead of one
    metric per question.

    Both RAW points are stored, and that's the record: a length and an angle
    are each recoverable from two points, but neither recovers the points, and
    which comparison you'll want isn't knowable at annotation time. length_px
    and angle_deg ride along BESIDE them as a convenience, because
    validation.metrics compares numbers rather than point pairs -- and because
    a length re-derived at each call site is a length two call sites eventually
    derive differently.

    labels -- the two point names, in click order.

    Returns {<first>, <second>, "length_px", "angle_deg"}, all None if skipped:

      <first>, <second> -- [x, y] in the CURRENT frame. Normally annotated with
                           no transforms applied, so they are original image
                           coordinates.
      length_px         -- distance between the two clicks.
      angle_deg         -- direction of the first->second vector, in image
                           coordinates (y down), so +90 means the second point
                           is directly BELOW the first.

    A metric's `unit` covers its whole value, so every key inherits "px_xy",
    which is not one of export.CONVERTIBLE_UNITS: length_px stays in pixels in
    a units="mm" export while body_length converts. That's the right default
    for a label whose job is grading a pipeline measured in pixels, and the key
    names carry their own units so the coarse parent tag can't mislead.
    """
    labels = [str(label) for label in labels]
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError(
            f"click_two_points needs two distinct labels, got {labels}"
        )

    return Metric("click_two_points", _click_two_points, {"labels": labels},
                  version="1", unit=unit, metric_name=name)


def _panel(segment):
    """Image, mask, and overlay side by side -- the standard annotation view."""
    image = np.asarray(segment.image)
    mask = segment.require_mask()
    return side_by_side(image, mask_to_bgr(mask), overlay_mask(image, mask))


def _wait_for_key(valid_keys):
    """Block (no timeout) until one of valid_keys is pressed, ignoring anything else."""
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


def _annotate_flags(segment):
    return _ask(segment, _panel(segment),
                "flag? (1=usable 2=not-an-organism 3=cut-off 4=multiple)",
                FLAG_KEYS)


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

    labels -- the two names the positions are stored under.
    points -- [(x0, y0), (x1, y1)] in the segment's current coordinates.
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
