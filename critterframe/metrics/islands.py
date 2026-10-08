"""Island metrics: n_islands, the disconnected fragments of a mask apart from the organism."""

import cv2
import numpy as np

from ..recipes import Metric


def n_islands(name=None, unit="count"):
    """Metric: how many disconnected fragments the mask has besides its largest component.

    Run it before `remove_islands()`, which leaves none to count.
    """
    return Metric("n_islands", _n_islands, version="1", unit=unit, metric_name=name)


def _n_islands(segment):
    mask = np.asarray(segment.require_mask()) > 0
    # The count includes the background as component 0, and the largest
    # foreground component is the organism, not an island.
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(0, count - 2)
