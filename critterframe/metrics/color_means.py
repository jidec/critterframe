"""
Mean colour and lightness of the masked organism, plus white-balanced colour and background colour.

Colour is only comparable across occurrences to the extent the imaging was, so calibrate or normalize before
comparing across sources; the last two are for uncontrolled photography, where there is nothing to calibrate against.
"""

import cv2
import numpy as np

from ..colorspaces import convert
from ..recipes import Metric
from .pixels import masked_pixels, visualize_mask


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


def white_balanced_color(name=None, unit="fraction"):
    """
    Metric: mean colour of the masked pixels after grey-world white balancing,
    as {"r", "g", "b"} on 0-1 scales.

    Grey-world assumes the WHOLE IMAGE averages to grey and scales each channel
    to make that true, then measures the organism under that correction. The
    assumption is crude and fails on an image that's genuinely mostly one
    colour, but it's computed from the image itself with no reference object,
    which is the only kind of correction available here.

    Uses the whole frame, not just the mask, to estimate the correction -- the
    organism is exactly the part whose colour you're trying to measure, so
    normalizing by it would define away the signal.
    """
    return Metric("white_balanced_color", _white_balanced_color, version="1",
                  unit=unit, metric_name=name)


def background_color(name=None, unit="fraction"):
    """
    Metric: mean colour and lightness of the pixels OUTSIDE the mask, as
    {"r", "g", "b", "lightness", "contrast"}.

    Background is information, not noise. What an organism was photographed
    against is context worth recording, and `contrast` -- the difference in
    lightness between the organism and its background -- is a direct QC signal:
    an organism close in tone to its background is one whose mask is most
    likely wrong, and one whose colour measurements are most likely
    contaminated by whatever the mask wrongly included.

    Returns lightness and contrast on 0-1 scales; contrast is signed, positive
    where the organism is lighter than its background.
    """
    return Metric("background_color", _background_color, version="1",
                  unit=unit, metric_name=name)


def _mean_lightness(segment):
    pixels = masked_pixels(segment)
    lightness = float(convert(pixels, "lab")[:, 0].mean() / 100.0)

    visualize_mask(segment, f"mean lightness {lightness:.3f}", "mean_lightness")
    return lightness


def _mean_color(segment):
    red, green, blue = convert(masked_pixels(segment), "rgb").mean(axis=0)
    return {"r": float(red), "g": float(green), "b": float(blue)}


def _grey_world_scale(image):
    """Per-channel gains that make the whole frame average to grey."""
    means = np.asarray(image).reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    return means.mean() / means


def _white_balanced_color(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    image = np.asarray(segment.image).astype(np.float32)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)

    balanced = np.clip(image * _grey_world_scale(image), 0, 255)
    blue, green, red = balanced[mask].mean(axis=0) / 255.0
    return {"r": float(red), "g": float(green), "b": float(blue)}


def _lightness(pixels):
    """Mean CIELAB lightness of an (N, 3) BGR array, on a 0-1 scale."""
    return float(convert(pixels.astype(np.uint8), "lab")[:, 0].mean() / 100.0)


def _background_color(segment):
    mask = segment.require_mask()
    background = ~mask
    if not background.any():
        raise ValueError("the mask covers the whole frame -- no background to measure")
    if not mask.any():
        raise ValueError("empty mask")

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    background_pixels = image[background]
    blue, green, red = background_pixels.mean(axis=0) / 255.0

    background_lightness = _lightness(background_pixels)
    organism_lightness = _lightness(image[mask])

    return {
        "r": float(red),
        "g": float(green),
        "b": float(blue),
        "lightness": background_lightness,
        "contrast": organism_lightness - background_lightness,
    }
