"""
Colour metrics: mean colour, hue/lightness fractions.

Measured over the masked organism only. Colour is only comparable across
occurrences to the extent the imaging was, so calibrate or normalize before
comparing across sources.
"""

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import annotate, side_by_side

# Hue ranges in OpenCV's 0-179 scale. Red wraps around the end of the scale, so
# it needs two bands rather than one -- the single most common way a hue filter
# quietly loses half its pixels.
HUE_BANDS = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 20)],
    "yellow": [(21, 35)],
    "green": [(36, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 160)],
}

# Floors below which a pixel has no meaningful hue: too grey to be any colour,
# or too dark to tell. Both on 0-1 scales.
MIN_SATURATION = 0.25
MIN_VALUE = 0.15

# Lightness below which a pixel counts as black, on a 0-1 scale.
BLACK_THRESHOLD = 0.20


def mean_lightness(name=None, unit="fraction"):
    """
    Metric: mean CIELAB lightness of the masked pixels, on a 0-1 scale.

    Lab's L rather than a plain RGB average because L is designed to track
    PERCEIVED lightness: a saturated yellow and a saturated blue of the same RGB
    mean are not equally light to an eye or a camera, and averaging channels
    would call them identical.
    """
    return Metric("mean_lightness", _mean_lightness, version="1", unit=unit,
                  metric_name=name)


def mean_color(name=None, unit="fraction"):
    """
    Metric: mean colour of the masked pixels, as {"r", "g", "b"} on 0-1 scales.

    The plainest possible colour summary, and the right one when what you want
    is comparable across a controlled imaging setup. It says nothing about
    PATTERN -- a black-and-white striped organism and a uniformly grey one
    return the same value -- so pair it with black_fraction or a colour cluster
    metric when pattern matters.
    """
    return Metric("mean_color", _mean_color, version="1", unit=unit,
                  metric_name=name)


def black_fraction(threshold=BLACK_THRESHOLD, name=None, unit="fraction"):
    """
    Metric: fraction of masked pixels darker than `threshold` lightness.

    Melanisation is the usual reason to want this, and it's a genuinely
    different question from mean lightness: a mostly-pale organism with heavy
    black markings and a uniformly mid-grey one can share a mean and differ
    completely here.

    threshold -- lightness cutoff on a 0-1 scale.
    """
    return Metric("black_fraction", _black_fraction, {"threshold": threshold},
                  version="1", unit=unit, metric_name=name)


def hue_fraction(hue, min_saturation=MIN_SATURATION, min_value=MIN_VALUE,
                 name=None, unit="fraction"):
    """
    Metric: fraction of masked pixels falling in one named hue band.

    hue            -- one of HUE_BANDS ("red", "yellow", "green", ...).
    min_saturation -- pixels greyer than this are excluded rather than assigned
                      a hue they don't really have.
    min_value      -- pixels darker than this are excluded for the same reason;
                      hue is meaningless in shadow.
    """
    if hue not in HUE_BANDS:
        raise ValueError(f"unknown hue {hue!r} -- expected one of {sorted(HUE_BANDS)}")

    return Metric(f"{hue}_fraction", _hue_fraction,
                  {"hue": hue, "min_saturation": min_saturation,
                   "min_value": min_value},
                  version="1", unit=unit, metric_name=name)


def red_fraction(name=None, **kwargs):
    """Metric: fraction of masked pixels in the red hue band. See hue_fraction()."""
    return hue_fraction("red", name=name, **kwargs)


def yellow_fraction(name=None, **kwargs):
    """Metric: fraction of masked pixels in the yellow hue band. See hue_fraction()."""
    return hue_fraction("yellow", name=name, **kwargs)


def _masked_pixels(segment):
    """
    The BGR pixel values under the mask, as an (N, 3) uint8 array.

    Every colour metric starts here, so the "only the organism's pixels count"
    rule is enforced once rather than re-implemented per metric.
    """
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image[mask]


def _mean_lightness(segment):
    pixels = _masked_pixels(segment)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    lightness = float(lab[:, 0, 0].mean() / 255.0)

    _visualize(segment, f"mean lightness {lightness:.3f}", "mean_lightness")
    return lightness


def _mean_color(segment):
    pixels = _masked_pixels(segment).astype(np.float32) / 255.0
    blue, green, red = pixels.mean(axis=0)
    return {"r": float(red), "g": float(green), "b": float(blue)}


def _black_fraction(segment, threshold=BLACK_THRESHOLD):
    pixels = _masked_pixels(segment)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    lightness = lab[:, 0, 0] / 255.0
    fraction = float((lightness < threshold).mean())

    _visualize(segment, f"black {fraction:.1%} (< {threshold})", "black_fraction")
    return fraction


def _hue_fraction(segment, hue, min_saturation=MIN_SATURATION, min_value=MIN_VALUE):
    pixels = _masked_pixels(segment)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0, :]
    hues = hsv[:, 0]
    saturation = hsv[:, 1] / 255.0
    value = hsv[:, 2] / 255.0

    colourful = (saturation >= min_saturation) & (value >= min_value)
    in_band = np.zeros(len(hues), dtype=bool)
    for low, high in HUE_BANDS[hue]:
        in_band |= (hues >= low) & (hues <= high)

    fraction = float((in_band & colourful).mean())
    _visualize(segment, f"{hue} {fraction:.1%}", f"{hue}_fraction")
    return fraction


def _visualize(segment, text, subdir):
    """
    The image beside the same image with everything outside the mask blanked --
    which makes it immediately obvious whether a surprising colour value is a
    real property of the organism or a mask that included the background.
    """
    if segment.panel_sink is None:
        return

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    masked = image.copy()
    masked[~segment.mask] = 0
    annotate(masked, text)

    segment.emit_panel(side_by_side(image, masked), subdir)
