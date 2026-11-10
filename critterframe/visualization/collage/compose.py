"""Compose: placed cutouts and decorations onto one canvas, in layers, with a blend mode."""

import math
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

MAX_PIXELS = 100_000_000
# `mean` keeps a float accumulator of the whole canvas, 16 bytes a pixel.
MAX_ACCUMULATED_PIXELS = 40_000_000
BLENDS = ("over", "mean", "multiply")
MARGIN = 0.25  # layout units kept clear around everything drawn
CACHE_BYTES = 256 * 1024 * 1024
MIN_TEXT_PX = 4


@dataclass
class Geometry:
    """Where everything lands on the canvas.

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        pieces: The placements, one row each in paste order, with `cx`, `cy` (the cutout's
            center on the canvas), `scale` (source pixels to canvas pixels) and `clip_px`.
        decorations: Decorations in canvas pixels.
        px_per_unit: Canvas pixels per layout unit.
    """

    width: int
    height: int
    pieces: pd.DataFrame
    decorations: list
    px_per_unit: float

    def spec(self):
        """Return the geometry, rounded, as JSON-serializable data: everything that decides the pixels."""
        pieces = [
            [
                str(row.occurrence_id),
                str(row.part),
                round(float(row.cx), 1),
                round(float(row.cy), 1),
                round(float(row.scale), 5),
                round(float(row.angle), 2),
                bool(row.flip),
                round(float(row.alpha), 3),
                row.clip,
                None if pd.isna(row.clip_px) else round(float(row.clip_px), 1),
                None if row.color is None else list(row.color),
            ]
            for row in self.pieces.itertuples(index=False)
        ]
        return {
            "size": [self.width, self.height],
            "pieces": pieces,
            "decorations": decoration_specs(self.decorations),
        }


def decoration_specs(decorations):
    """Return decorations as JSON-serializable data, an embedded image replaced by its digest."""
    return [{key: value for key, value in decoration.items() if key != "image"} for decoration in decorations]


def _rotated_box(width, height, angle):
    """Return the width and height of a `width` x `height` box turned by `angle` degrees."""
    radians = np.radians(angle)
    cos, sin = np.abs(np.cos(radians)), np.abs(np.sin(radians))
    return width * cos + height * sin, width * sin + height * cos


def _decoration_box(decoration):
    """Return `(left, top, right, bottom)` a decoration covers, in its own units."""
    kind = decoration["kind"]
    if kind == "line":
        points = np.asarray(decoration["points"], float)
        half = decoration["width"] / 2.0
        return (
            points[:, 0].min() - half,
            points[:, 1].min() - half,
            points[:, 0].max() + half,
            points[:, 1].max() + half,
        )
    if kind == "text":
        height = decoration["height"]
        width = 0.6 * height * len(decoration["text"])
        left = decoration["x"] - (width / 2.0 if decoration.get("align") == "center" else 0.0)
        top = decoration["y"] - (height / 2.0 if decoration.get("valign") == "center" else 0.0)
        return left, top, left + width, top + height
    return (
        decoration["x"],
        decoration["y"],
        decoration["x"] + decoration["width"],
        decoration["y"] + decoration["height"],
    )


def _scaled(decoration, factor, dx, dy):
    """Return a decoration in canvas pixels."""
    decoration = dict(decoration)
    if "points" in decoration:
        decoration["points"] = [[x * factor + dx, y * factor + dy] for x, y in decoration["points"]]
    if "x" in decoration:
        decoration["x"], decoration["y"] = decoration["x"] * factor + dx, decoration["y"] * factor + dy
    for key in ("height", "width"):
        if key in decoration:
            decoration[key] = decoration[key] * factor
    return decoration


def plan(placements, sizes, decorations, cell, max_side):
    """Return where every placement and decoration lands on a canvas fitted to `max_side`.

    Args:
        placements: The placements, in layout units.
        sizes: Each placement's cutout `(width, height)` in source pixels.
        decorations: Decorations in layout units.
        cell: Canvas pixels per layout unit, before fitting to `max_side`.
        max_side: The canvas's longest side at most, in pixels.

    Raises:
        ValueError: If the canvas would hold more than `MAX_PIXELS`.
    """
    pieces = placements.reset_index(drop=True).copy()
    sizes = np.asarray(sizes, float).reshape(-1, 2)
    longest, shortest = sizes.max(axis=1), sizes.min(axis=1)
    side = np.where(pieces["fit"].to_numpy() == "cover", shortest, longest)
    scale = pieces["size"].to_numpy(float) * cell / np.maximum(side, 1.0)
    box_w, box_h = _rotated_box(sizes[:, 0] * scale, sizes[:, 1] * scale, pieces["angle"].to_numpy(float))
    cx = pieces["x"].to_numpy(float) * cell
    cy = pieces["y"].to_numpy(float) * cell - np.where(
        pieces["anchor"].to_numpy() == "bottom", box_h / 2.0, 0.0
    )

    boxes = [np.column_stack([cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2])]
    boxes += [np.asarray([_decoration_box(_scaled(d, cell, 0.0, 0.0))]) for d in decorations]
    boxes = np.vstack([box for box in boxes if len(box)]) if len(pieces) or decorations else np.zeros((1, 4))
    margin = MARGIN * cell
    left, top = boxes[:, 0].min() - margin, boxes[:, 1].min() - margin
    right, bottom = boxes[:, 2].max() + margin, boxes[:, 3].max() + margin
    fit = min(1.0, max_side / max(right - left, bottom - top, 1.0))
    width, height = int(math.ceil((right - left) * fit)), int(math.ceil((bottom - top) * fit))
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"a {width}x{height} canvas is over MAX_PIXELS ({MAX_PIXELS:,}) -- lower max_side or cell"
        )

    pieces["cx"] = (cx - left) * fit
    pieces["cy"] = (cy - top) * fit
    pieces["scale"] = scale * fit
    pieces["clip_px"] = pieces["clip_size"].to_numpy(float) * cell * fit
    canvas_decorations = [_scaled(d, cell * fit, -left * fit, -top * fit) for d in decorations]
    return Geometry(width, height, pieces, canvas_decorations, cell * fit)


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------


def _clip_mask(height, width, clip, size):
    """Return a uint8 mask of a centered hexagon (pointy-top, `size` across its flats) or circle."""
    mask = np.zeros((height, width), np.uint8)
    cx, cy = width / 2.0, height / 2.0
    if clip == "circle":
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(size / 2.0)), 255, -1, cv2.LINE_AA)
        return mask
    radius = size / math.sqrt(3.0)
    points = [
        (cx + radius * math.cos(math.radians(a)), cy + radius * math.sin(math.radians(a)))
        for a in range(30, 390, 60)
    ]
    cv2.fillConvexPoly(mask, np.round(np.array(points) * 16).astype(np.int32), 255, cv2.LINE_AA, shift=4)
    return mask


def warp(bgra, scale, angle, flip, clip=None, clip_px=None):
    """Return a BGRA cutout scaled, flipped and turned, on a canvas just large enough to hold it.

    Args:
        bgra: The cutout.
        scale: Source pixels to output pixels.
        angle: Counterclockwise degrees.
        flip: Mirror it left to right first.
        clip: `"hex"` or `"circle"` to clip it to that shape, centered; None for none.
        clip_px: The clip shape's size in output pixels.
    """
    if flip:
        bgra = cv2.flip(bgra, 1)
    height, width = bgra.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # Premultiplied, so a transparent pixel's color never bleeds into its neighbors.
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    premultiplied = np.dstack([bgra[:, :, :3].astype(np.float32) * alpha, alpha[:, :, 0] * 255.0])
    resized = cv2.resize(premultiplied, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]

    if angle % 360.0:
        h, w = resized.shape[:2]
        out_w, out_h = (int(math.ceil(v)) for v in _rotated_box(w, h, angle))
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        matrix[0, 2] += out_w / 2.0 - w / 2.0
        matrix[1, 2] += out_h / 2.0 - h / 2.0
        resized = cv2.warpAffine(
            resized,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    out_alpha = np.clip(resized[:, :, 3], 0, 255)
    if clip is not None:
        out_alpha = out_alpha * (_clip_mask(*out_alpha.shape, clip, clip_px).astype(np.float32) / 255.0)
    safe = np.where(resized[:, :, 3:4] > 0, resized[:, :, 3:4] / 255.0, 1.0)
    color = np.clip(resized[:, :, :3] / safe, 0, 255)
    return np.dstack([color, out_alpha]).astype(np.uint8)


def _region(canvas_shape, piece_shape, cx, cy):
    """Return the canvas and piece slices where a piece centered at `(cx, cy)` overlaps the canvas."""
    height, width = piece_shape[:2]
    left, top = int(round(cx - width / 2.0)), int(round(cy - height / 2.0))
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + width, canvas_shape[1]), min(top + height, canvas_shape[0])
    if x0 >= x1 or y0 >= y1:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))


def paste_over(canvas, piece, cx, cy):
    """Composite a BGRA piece over a BGRA canvas, in place, centered at `(cx, cy)`."""
    region = _region(canvas.shape, piece.shape, cx, cy)
    if region is None:
        return
    target, source = region
    src = piece[source].astype(np.float32) / 255.0
    dst = canvas[target].astype(np.float32) / 255.0
    a, b = src[:, :, 3:4], dst[:, :, 3:4]
    out_alpha = a + b * (1.0 - a)
    safe = np.where(out_alpha > 0, out_alpha, 1.0)
    color = (src[:, :, :3] * a + dst[:, :, :3] * b * (1.0 - a)) / safe
    canvas[target] = np.clip(np.dstack([color, out_alpha]) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _paste_multiply(canvas, piece, cx, cy):
    """Multiply a BGRA piece into a BGRA canvas, in place: darkening only, as overlapping inks do."""
    region = _region(canvas.shape, piece.shape, cx, cy)
    if region is None:
        return
    target, source = region
    src = piece[source].astype(np.float32) / 255.0
    dst = canvas[target].astype(np.float32) / 255.0
    a = src[:, :, 3:4]
    color = dst[:, :, :3] * (1.0 - a + a * src[:, :, :3])
    out_alpha = a + dst[:, :, 3:4] * (1.0 - a)
    canvas[target] = np.clip(np.dstack([color, out_alpha]) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _accumulate(sums, weights, piece, cx, cy):
    """Add a BGRA piece's alpha-weighted color into the `mean` accumulators."""
    region = _region(sums.shape, piece.shape, cx, cy)
    if region is None:
        return
    target, source = region
    alpha = piece[source][:, :, 3].astype(np.float32) / 255.0
    sums[target] += piece[source][:, :, :3].astype(np.float32) * alpha[:, :, None]
    weights[target] += alpha


def text_image(content, height, color):
    """Return a line of text as a BGRA image about `height` pixels tall, on a transparent background.

    Args:
        content: The text.
        height: Pixel height of the line.
        color: BGR color.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100.0
    figure = Figure(dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    figure.patch.set_alpha(0.0)
    rgb = tuple(channel / 255.0 for channel in reversed(color))
    label = figure.text(0, 0, content, fontsize=height * 72.0 / dpi / 1.2, color=rgb, ha="left", va="bottom")
    box = label.get_window_extent(canvas.get_renderer())
    width_px, height_px = box.width + 4, box.height + 4
    figure.set_size_inches(width_px / dpi, height_px / dpi)
    label.set_position((2.0 / width_px, 2.0 / height_px))
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def _draw_decoration(canvas, decoration):
    """Draw one decoration, in canvas pixels, onto a BGRA canvas in place."""
    kind = decoration["kind"]
    color = tuple(int(c) for c in decoration.get("color", (128, 128, 128))) + (255,)
    if kind == "line":
        points = np.round(np.asarray(decoration["points"]) * 16).astype(np.int32)
        thickness = max(1, int(round(decoration["width"])))
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA, shift=4)
    elif kind == "rect":
        x0, y0 = int(round(decoration["x"])), int(round(decoration["y"]))
        x1 = int(round(decoration["x"] + decoration["width"]))
        y1 = int(round(decoration["y"] + decoration["height"]))
        cv2.rectangle(canvas, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, -1)
    elif kind == "text":
        if decoration["height"] < MIN_TEXT_PX:
            return
        image = text_image(decoration["text"], decoration["height"], decoration["color"])
        height, width = image.shape[:2]
        cx = decoration["x"] + (width / 2.0 if decoration.get("align") != "center" else 0.0)
        cy = decoration["y"] + (height / 2.0 if decoration.get("valign") != "center" else 0.0)
        paste_over(canvas, image, cx, cy)
    elif kind == "image":
        image = decoration["image"]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 3:
            image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
        size = (max(1, int(round(decoration["width"]))), max(1, int(round(decoration["height"]))))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        paste_over(canvas, image, decoration["x"] + size[0] / 2.0, decoration["y"] + size[1] / 2.0)
    else:
        raise ValueError(f"unknown decoration kind {kind!r}")


class _WarpCache:
    """Warped cutouts kept for reuse, newest last, within a byte budget."""

    def __init__(self, budget=CACHE_BYTES):
        self.budget = budget
        self.entries = OrderedDict()
        self.size = 0

    def get(self, key, build):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        value = build()
        self.entries[key] = value
        self.size += value.nbytes
        while self.size > self.budget and len(self.entries) > 1:
            _, dropped = self.entries.popitem(last=False)
            self.size -= dropped.nbytes
        return value


def draw(geometry, load, effects=(), blend="over", background=None):
    """Return the canvas: the under decorations, the cutouts in order, then the over decorations.

    Args:
        geometry: From `plan`.
        load: Called with a piece's row to return its source cutout as BGRA.
        effects: Effects applied to each warped cutout, in order.
        blend: `"over"` stacks cutouts, `"mean"` averages where they overlap (opacity follows
            how many overlap), `"multiply"` darkens as overlapping inks do.
        background: BGR fill; None leaves the canvas transparent.

    Returns:
        A BGRA array.
    """
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")
    if blend == "mean" and geometry.width * geometry.height > MAX_ACCUMULATED_PIXELS:
        raise ValueError(
            f"blend='mean' accumulates the whole canvas; {geometry.width}x{geometry.height} is over "
            f"{MAX_ACCUMULATED_PIXELS:,} pixels -- lower max_side"
        )
    canvas = np.zeros((geometry.height, geometry.width, 4), np.uint8)
    if background is not None:
        canvas[:, :, :3] = background
        canvas[:, :, 3] = 255
    elif blend == "multiply":
        canvas[:, :, :3] = 255

    for decoration in geometry.decorations:
        if decoration.get("layer") == "under":
            _draw_decoration(canvas, decoration)

    if blend == "mean":
        sums = np.zeros((geometry.height, geometry.width, 3), np.float32)
        weights = np.zeros((geometry.height, geometry.width), np.float32)

    cache = _WarpCache()
    for row in geometry.pieces.to_dict("records"):
        key = (row["occurrence_id"], row["part"])
        clip = row["clip"] if isinstance(row["clip"], str) else None
        clip_px = None if pd.isna(row["clip_px"]) else round(float(row["clip_px"]), 1)
        warp_key = (key, round(row["scale"], 5), round(row["angle"], 2), bool(row["flip"]), clip, clip_px)
        piece = cache.get(
            warp_key, lambda: warp(load(row), row["scale"], row["angle"], row["flip"], clip, clip_px)
        )
        if row["alpha"] < 1.0:
            piece = piece.copy()
            piece[:, :, 3] = (piece[:, :, 3].astype(np.float32) * row["alpha"]).astype(np.uint8)
        for effect in effects:
            piece = effect(piece, row)
        if blend == "over":
            paste_over(canvas, piece, row["cx"], row["cy"])
        elif blend == "multiply":
            _paste_multiply(canvas, piece, row["cx"], row["cy"])
        else:
            _accumulate(sums, weights, piece, row["cx"], row["cy"])

    if blend == "mean":
        covered = weights > 0
        mean = np.zeros((geometry.height, geometry.width, 4), np.uint8)
        mean[covered, :3] = np.clip(sums[covered] / weights[covered, None], 0, 255).astype(np.uint8)
        peak = float(weights.max()) if covered.any() else 1.0
        mean[:, :, 3] = np.clip(weights / peak * 255.0, 0, 255).astype(np.uint8)
        paste_over(canvas, mean, geometry.width / 2.0, geometry.height / 2.0)

    for decoration in geometry.decorations:
        if decoration.get("layer", "over") != "under":
            _draw_decoration(canvas, decoration)
    return canvas
