"""
Masked pixel access: the organism's pixels, and the panel that shows them.

Every colour metric starts from `masked_pixels`, so the "only the organism counts" rule lives in one place.
"""

import cv2
import numpy as np

from ..visualization.panels import annotate, side_by_side


def masked_pixels(segment, cap=None, seed=0, required=True):
    """
    The BGR pixel values under the mask, as an (N, 3) uint8 array.

    Every colour metric starts here, so the "only the organism's pixels count"
    rule is enforced once rather than re-implemented per metric.

    - `segment` -- the segment to read.
    - `cap` -- sample at most this many pixels, seeded so the same segment
      gives the same sample. None takes every masked pixel, which is what
      scoring wants; a cap is for POOLING many occurrences into one fit.
    - `seed` -- the sample's seed.
    - `required` -- False returns None instead of raising where there is no
      mask, for a caller pooling whatever it can get.
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
    """
    Emit the image beside the same image with everything outside the mask blanked.

    That makes it immediately obvious whether a surprising colour value is a real property of the organism or a mask
    that included the background.

    - `segment` -- the segment being measured.
    - `text` -- caption drawn on the masked half.
    - `subdir` -- the stage name the panel is emitted under.
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
