"""
Erode: pull a mask in from its edges, by a share of its own thickness or a fixed number of pixels.
"""

import cv2
import numpy as np

from ..maskops import edge_distance
from ..recipes import Transform
from ..visualization.panels import annotate

# The default share of the maximum inscribed radius to take off every edge.
DEFAULT_FRACTION = 0.1


def erode(fraction=DEFAULT_FRACTION, px=None):
    """
    Operation: shrink the working mask inward from every edge.

    Moves no pixels and never grows the mask; the image is left as it is. In a
    metric run it tightens the mask for that measurement only, and stored masks
    are untouched. A section thinner than twice the erosion is cut through:
    `n_components` in the info says how many pieces are left.

    - `fraction` -- erode by this share of the mask's maximum inscribed radius
      (half its thickness at the thickest point), so an elongated part and a
      round one lose the same share of their thickness. Between 0 and 1.
    - `px` -- erode by this many pixels instead; `fraction` is then ignored.
    """
    if px is not None:
        if not px > 0:
            raise ValueError(f"px must be positive, got {px!r}")
    elif not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    return Transform("erode", _erode, {"fraction": fraction, "px": px}, version="1")


def _erode(segment, fraction=DEFAULT_FRACTION, px=None):
    """Keep the pixels further than the erosion radius from the mask's edge."""
    original = segment.require_mask()
    area_before = int(original.sum())
    if area_before == 0:
        raise ValueError("empty mask")

    # The frame's edge counts as an edge of the mask: a mask touching it would
    # otherwise be eroded on every side but that one.
    distance = edge_distance(original)
    inscribed = float(distance.max())
    radius = float(px) if px is not None else fraction * inscribed

    eroded = distance > radius
    degenerate = not eroded.any()
    if degenerate:
        eroded = original

    area_after = int(eroded.sum())
    info = {
        "radius_px": radius,
        "inscribed_radius_px": inscribed,
        "area_before": area_before,
        "area_after": area_after,
        "removed_fraction": 1.0 - (area_after / area_before),
        "n_components": int(cv2.connectedComponents(eroded.astype(np.uint8),
                                                    connectivity=8)[0]) - 1,
        "degenerate": degenerate,
    }

    _visualize(segment, eroded, info)
    return segment.replace(mask=eroded), info


def _visualize(segment, eroded, info):
    """Retained mask in white, the ERODED rim in red, the same convention as remove_islands."""
    if segment.panel_sink is None:
        return

    original = segment.mask
    panel = np.zeros((*original.shape, 3), dtype=np.uint8)
    panel[eroded] = (255, 255, 255)
    panel[original & ~eroded] = (0, 0, 255)

    # Two short lines: a part crop is often narrower than one long line of text.
    annotate(panel, f"eroded {info['radius_px']:.1f}px"
                    + (" DEGENERATE" if info["degenerate"] else ""))
    annotate(panel, f"{info['removed_fraction']:.0%} of area", line=1)
    segment.emit_panel(panel, "erode")
