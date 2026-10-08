"""
Remove islands: disconnected fragments of a mask apart from the organism itself.
"""

import cv2
import numpy as np

from ..recipes import Transform
from ..visualization.panels import annotate


def remove_islands(min_area_frac=None):
    """
    Operation: drop connected components of the working mask that aren't the organism.

    Moves no pixels and never grows the mask; holes are left as they are.

    - `min_area_frac` -- None (default) keeps only the largest component. A
      fraction keeps every component at least that fraction of the largest's
      area, e.g. `0.1` for a wing that segmented as its own blob.
    """
    if min_area_frac is not None and not 0 < min_area_frac <= 1:
        raise ValueError(f"min_area_frac must be in (0, 1], got {min_area_frac!r}")
    return Transform("remove_islands", _remove_islands,
                     {"min_area_frac": min_area_frac}, version="1")


def _remove_islands(segment, min_area_frac=None):
    """
    Keep the largest 8-connected component, plus any at least `min_area_frac`
    of its area.
    """
    original = segment.require_mask()
    area_before = int(original.sum())
    if area_before == 0:
        raise ValueError("empty mask")

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        original.astype(np.uint8), connectivity=8)
    # Row 0 is the background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if min_area_frac is None:
        kept_labels = [1 + int(np.argmax(areas))]
    else:
        kept_labels = [1 + i for i, area in enumerate(areas)
                       if area >= min_area_frac * areas.max()]
    cleaned = np.isin(labels, kept_labels)

    area_after = int(cleaned.sum())
    info = {
        "area_before": area_before,
        "area_after": area_after,
        "removed_fraction": 1.0 - (area_after / area_before),
        "n_components": count - 1,
        "n_removed": count - 1 - len(kept_labels),
    }

    _visualize(segment, cleaned, info)
    return segment.replace(mask=cleaned), info


def _visualize(segment, cleaned, info):
    """Retained mask in white, REMOVED islands in red, the same convention as remove_appendages."""
    if segment.panel_sink is None:
        return

    original = segment.mask
    panel = np.zeros((*original.shape, 3), dtype=np.uint8)
    panel[cleaned] = (255, 255, 255)
    panel[original & ~cleaned] = (0, 0, 255)

    annotate(panel, f"removed {info['n_removed']}/{info['n_components']} comps, "
                    f"{info['removed_fraction']:.1%} "
                    f"({info['area_before']}->{info['area_after']}px)")
    segment.emit_panel(panel, "remove_islands")
