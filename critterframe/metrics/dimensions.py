"""Size and shape metrics: body_length, max_width, mask_area, bounding_box, elongation, jaggedness."""

import cv2
import numpy as np

from ..maskops import inscribed_radius, mask_bounds
from ..recipes import Metric
from ..visualization.panels import annotate, mask_to_bgr


def body_length(name=None, unit="px"):
    """Metric: the rows an oriented mask spans, from its topmost to its bottommost pixel.

    Run it after `orient()`: residual tilt inflates the value slightly.
    """
    return Metric("body_length", _body_length, version="1", unit=unit, metric_name=name)


def max_width(name=None, unit="px"):
    """Metric: the widest single row of an oriented mask, leftmost to rightmost pixel.

    A span, not a pixel count, so a row crossing two wings and the gap between them
    reports wingtip to wingtip. One noisy row can set it; the panel draws the width of
    every row.
    """
    return Metric("max_width", _max_width, version="1", unit=unit, metric_name=name)


def mask_area(name=None, unit="px2"):
    """Metric: the mask's pixel count."""
    return Metric("mask_area", _mask_area, version="1", unit=unit, metric_name=name)


def elongation(name=None, unit="ratio"):
    """Metric: how much longer the mask is than wide, from its pixels' spread along its principal axes.

    The square root of the ratio of the two variances: length over width for a rectangle,
    1 for a round mask. A leg, wing or stray speck pulls it toward 1.
    """
    return Metric("elongation", _elongation, version="1", unit=unit, metric_name=name)


# The default smoothing scale for jaggedness, as a share of the inscribed radius.
DEFAULT_SMOOTHING_FRACTION = 0.2


def jaggedness(fraction=DEFAULT_SMOOTHING_FRACTION, px=None, name=None, unit="ratio"):
    """Metric: the mask's perimeter over its perimeter once smoothed. 1.0 is a smooth edge.

    Appendages thinner than the smoothing scale count as roughness.

    Args:
        fraction: Smoothing scale as a share of the mask's maximum inscribed radius,
            between 0 and 1.
        px: Smoothing scale in pixels instead; `fraction` is then ignored.
        name: Name to store the value under.
        unit: Recorded unit.
    """
    if px is not None:
        if not px > 0:
            raise ValueError(f"px must be positive, got {px!r}")
    elif not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    return Metric(
        "jaggedness", _jaggedness, {"fraction": fraction, "px": px}, version="1", unit=unit, metric_name=name
    )


def bounding_box(name=None, unit="px"):
    """Metric: the mask's bounding box in the working frame, as `{"x", "y", "width", "height"}`.

    For the box in the original image, use `metrics.position.image_bounds`.
    """
    return Metric("bounding_box", _bounding_box, version="1", unit=unit, metric_name=name)


def _row_extent(row):
    """Return the horizontal span of mask pixels in one row, 0 if empty."""
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
        cv2.line(panel, (int(indices.min()), row), (int(indices.max()), row), (0, 255, 0), 1)
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
    """Return `(centroid_xy, variances, axes)` of a mask's pixels.

    Variances are largest first, each with a pixel's own 1/12 added; axes are unit
    vectors as columns.
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
    """Return the summed length of a mask's outer contours, and the contours."""
    contours, _hierarchy = cv2.findContours(
        np.asarray(mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return sum(cv2.arcLength(contour, True) for contour in contours), contours


def _jaggedness(segment, fraction=DEFAULT_SMOOTHING_FRACTION, px=None):
    mask = segment.require_mask()
    if mask.sum() < 2:
        raise ValueError("too few pixels to have an edge")

    sigma = max(1.0, float(px) if px is not None else fraction * inscribed_radius(mask))
    # Padded so the blur has room: clipped at the frame, a mask running off it
    # would be smoothed into a straight cut along the edge.
    pad = int(np.ceil(3 * sigma)) + 1
    padded = cv2.copyMakeBorder(mask.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
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
