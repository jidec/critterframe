"""
Appendage removal: strip thin legs and antennae from a mask while keeping the
body's original boundary intact.

Segmenters include appendages inconsistently -- the same specimen photographed
twice can come back with three legs one time and five the next -- so a body
measurement taken straight off a raw mask carries that inconsistency into the
trait. Removing them makes the measurement about the body.
"""

import logging

import cv2
import numpy as np

from ..recipes import Transform
from ..visualization.panels import annotate

logger = logging.getLogger(__name__)

# Appendage thickness threshold as a fraction of the mask's linear size.
# sqrt(area) converts pixel area to a linear measure, so the radius scales with
# specimen size: double the animal's linear dimensions (4x the area) and the
# radius doubles too. Structures thinner than 2*radius are treated as appendages.
#
# NOTE: this is a geometric proxy for physical thickness. Where a project has a
# real px/mm scale, prefer that -- it makes the threshold a physical quantity
# ("thinner than 0.4mm") instead of a relative one.
RELATIVE_RADIUS = 0.03

# Dilation is done at a slightly larger radius than the erosion, so the regrown
# core reliably covers the whole body before the intersection trims it back to
# the original boundary. Under-dilating would clip real body pixels.
DILATE_MARGIN = 1


def remove_appendages(relative_radius=RELATIVE_RADIUS):
    """
    Operation: strip thin appendages (legs, antennae) from the working mask.

    relative_radius -- appendage thickness threshold as a fraction of the
                       mask's linear size (see RELATIVE_RADIUS).
    """
    return Transform("remove_appendages", _remove_appendages,
                     {"relative_radius": relative_radius}, version="1")


def _disk(radius):
    """Circular structuring element -- isotropic, so no directional bias."""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_component(mask_uint8):
    """
    Keep only the biggest connected blob. After erosion this is the body core;
    anything else is a severed appendage or noise.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if n <= 1:
        return mask_uint8
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    return (labels == biggest).astype(np.uint8)


def _radius_for(area, relative_radius):
    """Erosion radius scaled to the mask's linear size, at least 1px."""
    return max(1, int(round(relative_radius * np.sqrt(area))))


def _remove_appendages(segment, relative_radius=RELATIVE_RADIUS):
    """
    Strip thin appendages from a segment's mask while keeping the body's
    ORIGINAL boundary intact.

    Opening alone would remove the appendages but also erode tapers and round
    corners, systematically shortening the body. Instead the opened result is
    used only as a SELECTOR: it decides which pixels are body, and the final
    intersection with the input restores their exact original edges.

      1. erode        -- appendages thinner than 2*radius vanish; their
                         connection to the body is severed
      2. largest blob -- keep the body core, discard severed fragments
      3. dilate back  -- regrow the core into a stencil that covers the body
                         (approximate boundary -- corners rounded, tapers blunt)
      4. intersect    -- AND with the original, so pixels keep their true edges
                         and anything outside the stencil (the appendages) is
                         dropped

    Doesn't move any pixels -- erode/dilate/keep-largest all operate on the
    same grid -- so the segment's mapping back to original coordinates is
    untouched and the image passes through unchanged.

    Masks with no appendages pass through essentially unchanged, which matters
    when the segmenter includes them inconsistently: otherwise cleaned and
    uncleaned masks would carry different distortions.
    """
    original = segment.require_mask().astype(np.uint8)
    area_before = int(original.sum())
    if area_before == 0:
        raise ValueError("empty mask")

    radius = _radius_for(area_before, relative_radius)
    eroded = cv2.erode(original, _disk(radius))

    if eroded.sum() == 0:
        # The whole mask was thinner than the kernel -- no body core to keep.
        # Return the original untouched rather than nothing.
        logger.warning("erosion removed everything (radius=%d); "
                       "returning mask unchanged", radius)
        cleaned = original.astype(bool)
        info = {"radius": radius, "area_before": area_before,
                "area_after": area_before, "removed_fraction": 0.0,
                "n_components": 1, "degenerate": True}
    else:
        core = _largest_component(eroded)
        n_components = int(
            cv2.connectedComponentsWithStats(eroded, connectivity=8)[0]) - 1

        stencil = cv2.dilate(core, _disk(radius + DILATE_MARGIN))
        cleaned = (stencil > 0) & (original > 0)

        area_after = int(cleaned.sum())
        info = {
            "radius": radius,
            "area_before": area_before,
            "area_after": area_after,
            "removed_fraction": 1.0 - (area_after / area_before),
            "n_components": n_components,
            "degenerate": False,
        }

    _visualize(segment, cleaned, info)
    return segment.replace(mask=cleaned), info


def _visualize(segment, cleaned, info):
    """
    Original mask in grey, retained body in white, REMOVED pixels in red -- so
    you can see exactly what was stripped and confirm it was appendages and not
    part of the body.
    """
    if segment.panel_sink is None:
        return

    original = segment.mask
    removed = original & ~cleaned

    panel = np.zeros((*original.shape, 3), dtype=np.uint8)
    panel[original] = (90, 90, 90)
    panel[cleaned] = (255, 255, 255)
    panel[removed] = (0, 0, 255)

    annotate(panel, f"r={info['radius']}px  removed {info['removed_fraction']:.1%} "
                    f"({info['area_before']}->{info['area_after']}px) "
                    f"comps {info['n_components']}")
    if info["degenerate"]:
        annotate(panel, "DEGENERATE - unchanged", line=1, color=(0, 0, 255))

    segment.emit_panel(panel, "remove_appendages")
