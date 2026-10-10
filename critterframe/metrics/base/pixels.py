"""Masked pixel access: the organism's pixels, and the panel that shows them."""

import cv2
import numpy as np

from ...visualization.panels import annotate, side_by_side


def masked_pixels(segment, cap=None, seed=0, required=True):
    """Return the BGR values of the pixels under the mask, as an `(N, 3)` uint8 array.

    Args:
        segment: The segment to read.
        cap: Sample at most this many pixels, for pooling many occurrences into one fit.
            None takes every masked pixel.
        seed: The sample's seed.
        required: False returns None where there is no mask, instead of raising.
    """
    mask = segment.mask if not required else segment.require_mask()
    if mask is None or not mask.any():
        if required:
            raise ValueError("empty mask")
        return None

    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    pixels = image[mask]
    if cap is not None and len(pixels) > cap:
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(len(pixels), cap, replace=False)]
    return pixels


def visualize_mask(segment, text, subdir):
    """Emit the image beside the same image with everything outside the mask blanked.

    Args:
        segment: The segment being measured.
        text: Caption drawn on the masked half.
        subdir: Stage name the panel is emitted under.
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
