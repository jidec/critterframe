"""
Automated QC metrics: blur, asymmetry, edge fraction, mask fraction.

Heuristics that flag likely-bad images and likely-bad masks. The WARN constants
are eyeballed defaults, not derived -- treat them as a sanity check and
calibrate real thresholds against human labels with validation.filters.
"""

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import annotate, diff_panel, mask_to_bgr

# Laplacian-variance floor below which a masked image is called blurry.
# Laplacian variance scales with image contrast and resolution, so this is
# specific to a camera/lens setup and should be recalibrated when that changes.
BLUR_WARN_VARIANCE = 50.0

# Above this asymmetry score (1 - IoU of a mask and its own mirror), a mask is
# too lopsided to trust -- usually a bad segmentation (a wing eaten on one side,
# background clutter picked up on the other) rather than a genuinely asymmetric
# organism.
ASYMMETRY_WARN_SCORE = 0.4

# Above this fraction of mask pixels lying on the image border, the organism is
# probably cut off by the frame -- which makes every length and area measurement
# an underestimate rather than a measurement.
EDGE_WARN_FRACTION = 0.02

# The same three constants keyed by the metric name each one belongs to, plus
# which side of it means "bad" -- what a caller needs to turn a metric into a
# filter without knowing which constant goes with which metric, or which way it
# points. validation.filters falls back to these for a metric whose sweep can't
# satisfy the constraint it was given, so an uncalibratable metric still filters
# on something rather than silently filtering on nothing.
#
# Keyed by metric NAME rather than by operation, because that's what survives
# into storage and into an export column; a metric renamed with name= is a
# deliberate reconfiguration and should say what its own threshold is.
WARN_THRESHOLDS = {
    "blur_variance": (BLUR_WARN_VARIANCE, "below"),
    "bilateral_asymmetry": (ASYMMETRY_WARN_SCORE, "above"),
    "edge_fraction": (EDGE_WARN_FRACTION, "above"),
}


def blur_variance(name=None, unit="laplacian_var"):
    """
    Metric: variance of the Laplacian over the masked region -- a standard
    focus-blur proxy, since sharp edges produce large second derivatives and a
    blurry image is muted everywhere.

    Restricted to mask pixels, so an out-of-focus or cluttered BACKGROUND can't
    drag the score down and a busy background can't fake a high one. The
    question is whether the ORGANISM is in focus.

    Rotation-invariant, so it can run before or after orient(). Higher is
    sharper. There's no universal scale -- compare across images from the same
    setup, or against BLUR_WARN_VARIANCE as a rough default cutoff.
    """
    return Metric("blur_variance", _blur_variance, version="1", unit=unit,
                  metric_name=name)


def bilateral_asymmetry(name=None, unit="fraction"):
    """
    Metric: how left-right asymmetric an ORIENTED mask is -- mirrors it across its
    own vertical centreline and returns 1 minus the IoU with its mirror.

    Most organisms are bilaterally symmetric viewed dorsally, so a high score
    usually means a bad segmentation rather than a lopsided specimen.

    Only meaningful once the body axis is vertical: on an unoriented mask the
    left/right split is arbitrary. A different question from orient()'s axis
    choice, which uses end-to-end asymmetry.

    Returns [0, 1]: 0.0 is perfectly symmetric, 1.0 is no overlap with its mirror.
    """
    return Metric("bilateral_asymmetry", _bilateral_asymmetry, version="1",
                  unit=unit, metric_name=name)


def edge_fraction(name=None, unit="fraction"):
    """
    Metric: fraction of mask pixels sitting on the image border -- how cut off
    the organism is by the edge of the frame.

    A cut-off organism is the failure case that most reliably corrupts size
    traits while looking perfectly fine as a segmentation: the mask is a correct
    outline of the visible part, and every length taken from it is an
    underestimate. Unlike blur or asymmetry, this doesn't degrade gradually --
    anything meaningfully above zero is worth excluding.

    Measured on the mask in ORIGINAL image coordinates, since "the edge of the
    frame" means the real frame, not the edge of a crop a recipe made.
    """
    return Metric("edge_fraction", _edge_fraction, version="1", unit=unit,
                  metric_name=name)


def mask_fraction(name=None, unit="fraction"):
    """
    Metric: fraction of the original image the mask covers.

    A blunt but effective catch for both segmentation failure directions: a mask
    covering 0.1% of the frame usually found a speck of dirt, and one covering
    80% usually found the substrate instead of the organism.
    """
    return Metric("mask_fraction", _mask_fraction, version="1", unit=unit,
                  metric_name=name)


def _to_gray(image):
    array = np.asarray(image)
    return array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


def _blur_variance(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    laplacian = cv2.Laplacian(_to_gray(segment.image), cv2.CV_64F)
    variance = float(laplacian[mask].var())

    if segment.panel_sink is not None:
        panel = cv2.normalize(np.abs(laplacian), None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel[~mask] = 0
        flag = "  BLURRY" if variance < BLUR_WARN_VARIANCE else ""
        annotate(panel, f"laplacian var {variance:.1f}{flag}")
        segment.emit_panel(panel, "blur_variance")

    return variance


def _bilateral_asymmetry(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    ys, xs = np.nonzero(mask)
    cx = xs.mean()

    mirrored = np.zeros_like(mask)
    source_x = np.round(2 * cx - xs).astype(int)
    valid = (source_x >= 0) & (source_x < mask.shape[1])
    mirrored[ys[valid], source_x[valid]] = True

    intersection = np.logical_and(mask, mirrored).sum()
    union = np.logical_or(mask, mirrored).sum()
    asymmetry = float(1.0 - intersection / union) if union else 1.0

    if segment.panel_sink is not None:
        panel = diff_panel(mask, mirrored)
        cv2.line(panel, (int(cx), 0), (int(cx), panel.shape[0]), (255, 128, 0), 1)
        flag = "  ASYMMETRIC" if asymmetry > ASYMMETRY_WARN_SCORE else ""
        annotate(panel, f"asymmetry {asymmetry:.2f}{flag}")
        segment.emit_panel(panel, "bilateral_asymmetry")

    return asymmetry


def _edge_fraction(segment):
    mask = segment.mask_in_original_coordinates()
    total = int(mask.sum())
    if total == 0:
        raise ValueError("empty mask")

    border = int(mask[0].sum() + mask[-1].sum()
                 + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    fraction = border / total

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        cv2.rectangle(panel, (0, 0), (mask.shape[1] - 1, mask.shape[0] - 1),
                      (0, 0, 255), 1)
        flag = "  CUT OFF" if fraction > EDGE_WARN_FRACTION else ""
        annotate(panel, f"edge {fraction:.1%} of {total}px{flag}")
        segment.emit_panel(panel, "edge_fraction")

    return fraction


def _mask_fraction(segment):
    mask = segment.mask_in_original_coordinates()
    if not mask.any():
        raise ValueError("empty mask")
    return float(mask.sum() / mask.size)
