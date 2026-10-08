"""
Size and shape metrics: body_length, max_width, mask_area, bounding_box, elongation, jaggedness.

Run body_length and max_width on ORIENTED masks: they measure image axes, and
only orientation makes an image axis correspond to the organism's own.
elongation and jaggedness need no orientation.
"""

import cv2
import numpy as np

from ..maskops import inscribed_radius, mask_bounds
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


def elongation(name=None, unit="ratio"):
    """
    Metric: how much longer the mask is than it is wide, from the spread of its
    pixels along its two principal axes.

    The square root of the ratio of those two variances, which is exactly
    length / width for a rectangle and the axis ratio for an ellipse. It needs
    no orientation; 1 is round. Pixels far from the centre weigh by their
    squared distance, so a leg, a wing or a stray speck pulls it toward round:
    run it after `remove_islands()` / `remove_appendages()` for the body's own
    shape, or before them where a pulled value is the signal (a leaking mask).
    """
    return Metric("elongation", _elongation, version="1", unit=unit, metric_name=name)


# The default smoothing scale for jaggedness, as a share of the inscribed radius.
DEFAULT_SMOOTHING_FRACTION = 0.2


def jaggedness(fraction=DEFAULT_SMOOTHING_FRACTION, px=None, name=None, unit="ratio"):
    """
    Metric: how much edge the mask loses when its outline is smoothed. 1.0 is
    smooth; a toothed or ragged edge is higher.

    The mask's perimeter over the perimeter of the same mask smoothed at a small
    scale: teeth and small notches go, the overall shape stays, so a smoothly
    curved part reads as smooth. A raster edge's staircase is on both sides and
    cancels. Thin appendages are narrower than the scale and count as
    roughness: run it after `remove_appendages()` for the body's own edge.

    - `fraction` -- the smoothing scale as a share of the mask's maximum
      inscribed radius (half its thickness), so "fine" is relative to the
      part. Between 0 and 1.
    - `px` -- the smoothing scale in pixels instead; `fraction` is then ignored.
    """
    if px is not None:
        if not px > 0:
            raise ValueError(f"px must be positive, got {px!r}")
    elif not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    return Metric("jaggedness", _jaggedness, {"fraction": fraction, "px": px},
                  version="1", unit=unit, metric_name=name)


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


def _principal_axes(mask):
    """
    `(centroid_xy, variances, axes)` of a mask's pixels: variances largest first,
    each with a pixel's own 1/12 added, and the matching unit axes as columns.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        raise ValueError("too few pixels to have a shape")
    coordinates = np.stack([xs, ys]).astype(np.float64)
    # A pixel is a unit square, not a point: its own spread is 1/12 per axis.
    # Without it a mask one pixel wide has zero width and an infinite ratio.
    covariance = np.cov(coordinates, bias=True) + np.eye(2) / 12.0
    variances, axes = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    return coordinates.mean(axis=1), variances[order], axes[:, order]


def _elongation(segment):
    mask = segment.require_mask()
    centroid, variances, axes = _principal_axes(mask)
    value = float(np.sqrt(variances[0] / variances[1]))

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        for variance, axis, color in zip(variances, axes.T, ((0, 255, 0), (0, 0, 255))):
            reach = 2 * np.sqrt(variance) * axis
            start = tuple(int(round(v)) for v in centroid - reach)
            end = tuple(int(round(v)) for v in centroid + reach)
            cv2.line(panel, start, end, color, 1)
        annotate(panel, f"elongation {value:.2f}")
        segment.emit_panel(panel, "elongation")
    return value


def _outer_perimeter(mask):
    """The summed length of a mask's outer contours; holes are not counted."""
    contours, _hierarchy = cv2.findContours(np.asarray(mask).astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(contour, True) for contour in contours), contours


def _jaggedness(segment, fraction=DEFAULT_SMOOTHING_FRACTION, px=None):
    mask = segment.require_mask()
    if mask.sum() < 2:
        raise ValueError("too few pixels to have an edge")

    sigma = max(1.0, float(px) if px is not None else fraction * inscribed_radius(mask))
    # Padded so the blur has room: clipped at the frame, a mask running off it
    # would be smoothed into a straight cut along the edge.
    pad = int(np.ceil(3 * sigma)) + 1
    padded = cv2.copyMakeBorder(mask.astype(np.float32), pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    smoothed = cv2.GaussianBlur(padded, (0, 0), sigma) > 0.5
    if not smoothed.any():
        raise ValueError("too small for the smoothing scale")

    perimeter, _contours = _outer_perimeter(mask)
    smoothed_perimeter, smoothed_contours = _outer_perimeter(smoothed)
    value = float(perimeter / smoothed_perimeter)

    if segment.panel_sink is not None:
        panel = mask_to_bgr(mask)
        shifted = [contour - pad for contour in smoothed_contours]
        cv2.drawContours(panel, shifted, -1, (0, 0, 255), 1)
        annotate(panel, f"jaggedness {value:.2f} at {sigma:.1f}px")
        segment.emit_panel(panel, "jaggedness")
    return value


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
    return mask_bounds(segment.require_mask())


# sketch-friendly alias: the doc's Antenna pipeline writes body_length(), other
# pipelines write length(). Same operation and the same recipe hash either way.
length = body_length
