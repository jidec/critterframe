"""Mean color metrics: mean_lightness, mean_color, white_balanced_color, background_color."""

import cv2
import numpy as np

from ..colorspaces import convert
from ..recipes import Metric
from .pixels import masked_pixels, visualize_mask


def mean_lightness(name=None, unit="fraction"):
    """Metric: mean CIELAB lightness of the masked pixels, on a 0-1 scale."""
    return Metric("mean_lightness", _mean_lightness, version="1", unit=unit, metric_name=name)


def mean_color(name=None, unit="fraction"):
    """Metric: mean color of the masked pixels, as `{"r", "g", "b"}` on 0-1 scales."""
    return Metric("mean_color", _mean_color, version="1", unit=unit, metric_name=name)


def white_balanced_color(name=None, unit="fraction"):
    """Metric: mean color of the masked pixels after gray-world white balancing, as `{"r", "g", "b"}`.

    The correction is estimated from the whole frame, not the mask, and assumes the frame
    averages to gray.
    """
    return Metric("white_balanced_color", _white_balanced_color, version="1", unit=unit, metric_name=name)


def background_color(name=None, unit="fraction"):
    """Metric: mean color and lightness of the pixels outside the mask.

    Returns:
        The metric. Its value is `{"r", "g", "b", "lightness", "contrast"}` on 0-1 scales.
        `contrast` is the organism's lightness minus the background's.
    """
    return Metric("background_color", _background_color, version="1", unit=unit, metric_name=name)


def _mean_lightness(segment):
    pixels = masked_pixels(segment)
    lightness = float(convert(pixels, "lab")[:, 0].mean() / 100.0)

    visualize_mask(segment, f"mean lightness {lightness:.3f}", "mean_lightness")
    return lightness


def _mean_color(segment):
    red, green, blue = convert(masked_pixels(segment), "rgb").mean(axis=0)
    return {"r": float(red), "g": float(green), "b": float(blue)}


def _grey_world_scale(image):
    """Return the per-channel gains that make the whole frame average to gray."""
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
    """Return the mean CIELAB lightness of an `(N, 3)` BGR array, on a 0-1 scale."""
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
