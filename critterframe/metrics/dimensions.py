"""
Size and shape metrics: length, width, area.

body_length and max_width both assume an ORIENTED segment -- they measure
extents along the image axes, which only correspond to the organism's own axes
once transforms.orient.orient() has put the body axis vertical. Put orient() in
the recipe's transforms before either of them. mask_area doesn't care, since a
pixel count is rotation-invariant.

Values are in pixels. Converting to millimetres needs a per-image scale, which
is a project-specific calibration rather than something a shape metric can know
-- see calibration.scale for one way to recover
it. `length` is available as an alias for `body_length`.
"""

import cv2
import numpy as np

from ..recipes import Metric
from ..visualization.panels import annotate, mask_to_bgr


def body_length(name=None, unit="px"):
    """
    Metric: vertical extent of an ORIENTED mask -- the number of rows spanned
    from the topmost to the bottommost mask pixel.

    Assumes orientation put the body axis exactly vertical. A few degrees of
    residual tilt inflates this slightly (you're measuring the bounding-box
    height of a tilted object), a small systematic upward bias.
    """
    return Metric("body_length", _body_length, version="1", unit=unit,
                  metric_name=name)


def max_width(name=None, unit="px"):
    """
    Metric: maximum horizontal span across an ORIENTED mask -- the width of the
    widest single row.

    Width is the SPAN from leftmost to rightmost mask pixel in the row, not the
    count of mask pixels. On a spread-wing specimen a row crosses left wing,
    background gap, then right wing; the span correctly reports wingtip to
    wingtip, whereas a pixel count would omit the gap.

    This is an extremum, so one noisy row can define it. The visualization draws
    the full per-row width profile alongside the mask -- a sustained peak is a
    real measurement, a single spike is noise.
    """
    return Metric("max_width", _max_width, version="1", unit=unit,
                  metric_name=name)


def mask_area(name=None, unit="px2"):
    """
    Metric: pixel count of the mask -- how much image the segmented part
    occupies.

    Unlike body_length/max_width this doesn't need an oriented mask, so it can
    run before or after orient(). It's also the natural size metric for parts
    whose "length" isn't well defined, like a wing.
    """
    return Metric("mask_area", _mask_area, version="1", unit=unit,
                  metric_name=name)


def bounding_box(name=None, unit="px"):
    """
    Metric: the mask's bounding box in the CURRENT frame, as
    {"x", "y", "width", "height"}.

    Reported as four numbers in one metric rather than four metrics because
    they're only meaningful together -- and because a caller wanting just the
    height should use body_length(), which says what it means.

    Note this is the box in whatever frame the transforms left behind, not in
    original image coordinates; for position in the original image, use
    metrics.position.
    """
    return Metric("bounding_box", _bounding_box, version="1", unit=unit,
                  metric_name=name)


def _row_extent(row):
    """Horizontal span of mask pixels in a single row, 0 if empty."""
    indices = np.nonzero(row)[0]
    return int(indices.max() - indices.min() + 1) if len(indices) else 0


def _body_length(segment):
    mask = segment.require_mask()
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        raise ValueError("empty mask")

    y0, y1 = int(ys.min()), int(ys.max())
    length = y1 - y0 + 1

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        width = panel.shape[1]
        cv2.line(panel, (0, y0), (width, y0), (0, 255, 0), 1)
        cv2.line(panel, (0, y1), (width, y1), (0, 255, 0), 1)
        cv2.arrowedLine(panel, (5, y0), (5, y1), (0, 255, 255), 1, tipLength=0.03)
        annotate(panel, f"length {length}px")
        segment.emit_panel(panel, "body_length")

    return length


def _max_width(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    widths = np.array([_row_extent(mask[y]) for y in range(mask.shape[0])])
    width = int(widths.max())
    row = int(widths.argmax())

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        indices = np.nonzero(mask[row])[0]
        cv2.line(panel, (int(indices.min()), row), (int(indices.max()), row),
                 (0, 255, 0), 1)
        annotate(panel, f"max width {width}px @row {row}")

        # width profile strip: each row's width as a horizontal bar, so a lone
        # spike (noise) is instantly distinguishable from a sustained peak
        strip_width = 60
        strip = np.zeros((panel.shape[0], strip_width, 3), dtype=np.uint8)
        if width > 0:
            for y, value in enumerate(widths):
                n = int(round(value / width * (strip_width - 1)))
                if n > 0:
                    strip[y, :n] = (0, 255, 0) if y == row else (140, 140, 140)
        panel = np.hstack([panel, strip])

        segment.emit_panel(panel, "max_width")

    return width


def _mask_area(segment):
    mask = segment.require_mask()
    if not mask.any():
        raise ValueError("empty mask")

    area = int(mask.sum())

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        annotate(panel, f"area {area}px")
        segment.emit_panel(panel, "mask_area")

    return area


def _bounding_box(segment):
    mask = segment.require_mask()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


# sketch-friendly alias: the doc's Antenna pipeline writes body_length(), other
# pipelines write length(). Same operation and the same recipe hash either way.
length = body_length
