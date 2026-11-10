"""Layouts: an arrangement into slots, the positions items are pasted at, in layout units.

One layout unit is one cell. y increases downward and an angle is counterclockwise in degrees, as
on screen. A slot's `size` is the side an item of scale 1 is fitted to: its longest side for
`fit="contain"`, its shortest for `fit="cover"`.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assign import match, sequence
from .base import Step, array_digest, read_image

GOLDEN_ANGLE = 180.0 * (3.0 - math.sqrt(5.0))
ROTATIONS = ("none", "radial", "tangent")
MAX_CIRCLES = 1000
DECORATION_COLOR = (128, 128, 128)

_SLOT_DEFAULTS = {
    "angle": 0.0,
    "size": 1.0,
    "flip": False,
    "anchor": "center",
    "fit": "contain",
    "clip": None,
    "clip_size": np.nan,
    "alpha": 1.0,
    "group": None,
    "color": None,
    "item": -1,
}


@dataclass
class Placed:
    """What a layout returns.

    Attributes:
        slots: One row per slot: `x, y, angle, size, flip, anchor, fit, clip, clip_size, alpha,
            source, group, color, item`. Slots sharing a `source` receive the same item; an
            `item` of -1 leaves the choice to the assignment.
        decorations: Layout decorations in layout units (see `decorate`).
        targets: Each item's target position in layout units, for matching by position.
        slot_values: Each slot's features, for matching by value.
        default_assign: The assignment used when the collage names none.
    """

    slots: pd.DataFrame
    decorations: list = field(default_factory=list)
    targets: np.ndarray = None
    slot_values: np.ndarray = None
    default_assign: object = None


def make_slots(xs, ys, **columns):
    """Return a slots frame at these positions, every other column at its default.

    Args:
        xs: Horizontal positions.
        ys: Vertical positions.
        **columns: Per-slot values, or one value for every slot, overriding the defaults.
    """
    xs = np.asarray(xs, float).ravel()
    frame = {"x": xs, "y": np.asarray(ys, float).ravel()}
    for name, default in {**_SLOT_DEFAULTS, "source": np.arange(len(xs)), **columns}.items():
        if isinstance(default, (list, np.ndarray, pd.Series)):
            frame[name] = list(default)
        else:
            frame[name] = [default] * len(xs)
    slots = pd.DataFrame(frame)
    slots["source"] = slots["source"].astype(int)
    slots["item"] = slots["item"].astype(int)
    return slots


def text(x, y, content, height, color=DECORATION_COLOR, align="center", valign="top", layer="over"):
    """Return a text decoration, in layout units."""
    return {
        "kind": "text",
        "x": float(x),
        "y": float(y),
        "text": str(content),
        "height": float(height),
        "color": list(color),
        "align": align,
        "valign": valign,
        "layer": layer,
    }


def line(points, width, color=DECORATION_COLOR, layer="under"):
    """Return a polyline decoration, in layout units."""
    return {
        "kind": "line",
        "points": [[float(x), float(y)] for x, y in points],
        "width": float(width),
        "color": list(color),
        "layer": layer,
    }


class Layout(Step):
    """A step from an arrangement to slots.

    Args:
        name: As in `Step`.
        function: Called as `function(arrangement, **parameters)`, returning a `Placed`.
        parameters: As in `Step`; each value JSON-serializable.
        data: Non-serializable inputs passed to `function` beside `parameters`, such as an image,
            each represented in `parameters` by its digest.
    """

    kind = "layout"

    def __init__(self, name, function, parameters=None, data=None):
        super().__init__(name, parameters)
        self.function = function
        self.data = dict(data or {})

    def place(self, arrangement):
        """Return the slots for this arrangement."""
        if len(arrangement) == 0:
            return Placed(make_slots([], []), default_assign=sequence())
        return self.function(arrangement, **self.parameters, **self.data)


def _pitch(arrangement, spacing):
    """Return the distance between slot centers that keeps the largest item clear of its neighbors."""
    return float(spacing) * max(float(np.max(arrangement.scales())), 1e-6)


def _check_rotate(rotate):
    if rotate not in ROTATIONS:
        raise ValueError(f"rotate must be one of {ROTATIONS}, got {rotate!r}")


def _rotation(phi, rotate):
    """Return the angle that turns an item's top outward (`radial`) or along the curve (`tangent`).

    Args:
        phi: The slot's direction from the center, in screen degrees (clockwise, y down).
        rotate: One of `ROTATIONS`.
    """
    phi = np.asarray(phi, float)
    if rotate == "radial":
        return -(phi + 90.0) % 360.0
    if rotate == "tangent":
        return -(phi + 180.0) % 360.0
    return np.zeros_like(phi)


# ---------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------


def grid(columns=None, spacing=1.0, overlap=0.0, anchor="center"):
    """Layout: rows of slots, left to right and top to bottom.

    Args:
        columns: Slots per row; about square if None.
        spacing: Slot pitch, as a multiple of the largest item.
        overlap: Fraction of the pitch neighbors overlap by, or `(horizontal, vertical)`.
        anchor: `"center"`, or `"bottom"` to stand every item on its row's baseline.
    """
    if anchor not in ("center", "bottom"):
        raise ValueError(f"anchor must be 'center' or 'bottom', got {anchor!r}")
    overlap = list(overlap) if isinstance(overlap, (list, tuple)) else [overlap, overlap]
    return Layout(
        "grid", _grid, {"columns": columns, "spacing": spacing, "overlap": overlap, "anchor": anchor}
    )


def _grid(arrangement, columns, spacing, overlap, anchor):
    n = len(arrangement)
    columns = int(columns) if columns else max(1, math.ceil(math.sqrt(n)))
    pitch = _pitch(arrangement, spacing)
    index = np.arange(n)
    xs = (index % columns) * pitch * (1.0 - overlap[0])
    ys = (index // columns) * pitch * (1.0 - overlap[1])
    if anchor == "bottom":
        ys = ys + pitch / 2.0
    return Placed(make_slots(xs, ys, size=float(spacing), anchor=anchor), default_assign=sequence())


def _hex_lattice(columns, rows, pitch):
    """Return pointy-top hexagonal lattice points, odd rows shifted right by half a pitch."""
    row, column = np.divmod(np.arange(columns * rows), columns)
    xs = column * pitch + (row % 2) * pitch / 2.0
    ys = row * pitch * math.sqrt(3.0) / 2.0
    return xs, ys


def hexgrid(columns=None, spacing=1.0, clip=True):
    """Layout: a hexagonal lattice, each item optionally clipped to its hexagon so the tiles meet.

    With `clip`, each item covers its hexagon, so opaque patches (from `crop_inside_mask`)
    tile without gaps.

    Args:
        columns: Slots per row; about square if None.
        spacing: Distance between neighbors, as a multiple of the largest item.
        clip: Clip each item to its hexagon.
    """
    return Layout("hexgrid", _hexgrid, {"columns": columns, "spacing": spacing, "clip": bool(clip)})


def _hexgrid(arrangement, columns, spacing, clip):
    n = len(arrangement)
    columns = int(columns) if columns else max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / columns)
    pitch = _pitch(arrangement, spacing)
    xs, ys = _hex_lattice(columns, rows, pitch)
    if not clip:
        return Placed(make_slots(xs[:n], ys[:n], size=float(spacing)), default_assign=sequence())
    # A pointy-top hexagon one pitch wide is 2/sqrt(3) pitches tall; covering that height
    # leaves no corner of it empty.
    scale = pitch / float(spacing)
    slots = make_slots(
        xs[:n],
        ys[:n],
        size=float(spacing) * 2.0 / math.sqrt(3.0),
        fit="cover",
        clip="hex",
        clip_size=pitch / scale,
    )
    return Placed(slots, default_assign=sequence())


def mosaic(image, columns=40, spacing=1.0):
    """Layout: a grid of slots, each carrying the mean color of its block of a target image.

    Its default assignment matches each slot to the item nearest its color, reusing items, so
    the items need values from `match_on` in RGB 0-1, as `mean_color()` stores them.

    Args:
        image: The target image, an array or a path.
        columns: Slots across; rows follow from the image's aspect ratio.
        spacing: Slot pitch, as a multiple of the largest item.
    """
    array = read_image(image)
    if array.ndim == 2:
        array = np.dstack([array] * 3)
    array = np.ascontiguousarray(array[:, :, :3])
    return Layout(
        "mosaic",
        _mosaic,
        {"image": array_digest(array), "columns": int(columns), "spacing": spacing},
        data={"array": array},
    )


def _mosaic(arrangement, image, columns, spacing, array):
    import cv2

    height, width = array.shape[:2]
    rows = max(1, int(round(columns * height / width)))
    blocks = cv2.resize(array, (columns, rows), interpolation=cv2.INTER_AREA)
    rgb = blocks.reshape(-1, 3)[:, ::-1].astype(float) / 255.0
    pitch = _pitch(arrangement, spacing)
    index = np.arange(columns * rows)
    slots = make_slots(
        (index % columns) * pitch,
        (index // columns) * pitch,
        size=float(spacing),
        color=[tuple(round(float(c), 4) for c in value) for value in rgb],
    )
    return Placed(slots, slot_values=rgb, default_assign=match(on="values", reuse=True))


def _shape_mask(shape):
    """Return a shape given as a mask, an image or a path, as a boolean mask."""
    array = read_image(shape)
    if array.dtype == bool or array.ndim == 2:
        return np.asarray(array) > 0
    if array.shape[2] == 4:
        return array[:, :, 3] > 0
    return (array > 0).any(axis=2)


def within(shape, spacing=1.0):
    """Layout: a hexagonal lattice clipped to a shape, filled from its center outward.

    The lattice is as coarse as still gives every item a slot, so the shape is filled almost
    exactly; a few slots at its edge may stay empty.

    Args:
        shape: A mask, an image (its alpha, or its nonzero pixels) or a path to one, e.g. one
            specimen's own cutout.
        spacing: Slot pitch, as a multiple of the largest item.
    """
    mask = _shape_mask(shape)
    if not mask.any():
        raise ValueError("within() was given an empty shape")
    return Layout(
        "within",
        _within,
        {"shape": array_digest(mask), "spacing": spacing},
        data={"mask": mask},
    )


def _lattice_in(mask, step):
    """Return the hexagonal lattice points with this step that land inside the mask, in pixels."""
    height, width = mask.shape
    columns = int(width / step) + 2
    rows = int(height / (step * math.sqrt(3.0) / 2.0)) + 2
    xs, ys = _hex_lattice(columns, rows, step)
    xs, ys = xs + step / 2.0, ys + step / 2.0
    inside = (xs < width) & (ys < height)
    xs, ys = xs[inside], ys[inside]
    keep = mask[ys.astype(int), xs.astype(int)]
    return xs[keep], ys[keep]


def _within(arrangement, shape, spacing, mask):
    n = len(arrangement)
    step = math.sqrt(mask.sum() / (n * math.sqrt(3.0) / 2.0))
    xs, ys = _lattice_in(mask, step)
    while len(xs) < n and step > 0.5:
        step *= 0.97
        xs, ys = _lattice_in(mask, step)
    if len(xs) < n:
        raise ValueError(
            f"the shape holds {len(xs)} slots at most, fewer than {n} items -- pass a larger one"
        )
    center_y, center_x = np.argwhere(mask).mean(axis=0)
    distance = np.hypot(xs - center_x, ys - center_y)
    order = np.lexsort((xs, ys, np.round(distance, 6)))
    pitch = _pitch(arrangement, spacing)
    return Placed(
        make_slots(xs[order] / step * pitch, ys[order] / step * pitch, size=float(spacing)),
        default_assign=sequence(),
    )


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def spiral(spacing=1.0, coil=1.0, rotate="tangent"):
    """Layout: an Archimedean spiral outward from the center, items evenly spaced along it.

    Args:
        spacing: Distance between neighbors along the curve, as a multiple of the largest item.
        coil: Distance between successive turns, as a multiple of `spacing`.
        rotate: `"tangent"` turns each item along the curve, `"radial"` points it outward,
            `"none"` keeps it upright.
    """
    _check_rotate(rotate)
    return Layout("spiral", _spiral, {"spacing": spacing, "coil": coil, "rotate": rotate})


def _spiral(arrangement, spacing, coil, rotate):
    pitch = _pitch(arrangement, spacing)
    b = pitch * coil / (2.0 * math.pi)
    thetas = [0.0]
    for _ in range(len(arrangement) - 1):
        theta = thetas[-1]
        thetas.append(theta + pitch / math.hypot(b * theta, b))
    thetas = np.asarray(thetas)
    radii = b * thetas
    angles = _rotation(np.degrees(thetas), rotate)
    slots = make_slots(
        radii * np.cos(thetas), radii * np.sin(thetas), size=float(spacing), angle=list(angles)
    )
    return Placed(slots, default_assign=sequence())


def rings(spacing=1.0, by_group=False, symmetry=None, mirror=False, rotate="radial", center=True):
    """Layout: concentric rings, each holding as many items as fit around it.

    Args:
        spacing: Distance between neighbors and between rings, as a multiple of the largest item.
        by_group: Start each group (outermost label) on a ring of its own.
        symmetry: Repeat each ring's items this many times around it, as a kaleidoscope.
        mirror: With `symmetry`, mirror every other repeat.
        rotate: As in `spiral`.
        center: Put the first item at the center.
    """
    _check_rotate(rotate)
    if symmetry is not None and (by_group or int(symmetry) < 1):
        raise ValueError("symmetry= needs a positive count and can't be combined with by_group=True")
    return Layout(
        "rings",
        _rings,
        {
            "spacing": spacing,
            "by_group": bool(by_group),
            "symmetry": None if symmetry is None else int(symmetry),
            "mirror": bool(mirror),
            "rotate": rotate,
            "center": bool(center),
        },
    )


def _ring_capacity(ring):
    """Return how many items fit around ring `ring`, one pitch apart."""
    return max(1, int(2.0 * math.pi * ring))


def _rings(arrangement, spacing, by_group, symmetry, mirror, rotate, center):
    n = len(arrangement)
    pitch = _pitch(arrangement, spacing)
    rows = []  # (ring, phi, source, flip, group)

    def ring_slot(ring, phi, source, flip=False, group=None):
        rows.append((ring, phi, source, flip, group))

    if by_group:
        if arrangement.groups is None:
            raise ValueError("rings(by_group=True) needs groups: arrange with group_by() first")
        labels = [group[0] for group in arrangement.groups]
        counts = []
        for label in labels:
            if counts and counts[-1][0] == label:
                counts[-1][1] += 1
            else:
                counts.append([label, 1])
        ring, source = 0, 0
        for label, count in counts:
            while count:
                ring += 1
                here = min(count, _ring_capacity(ring))
                for j in range(here):
                    ring_slot(ring, 360.0 * j / here, source, group=label)
                    source += 1
                count -= here
    else:
        source, ring = 0, 0
        if center and symmetry is None:
            ring_slot(0, 0.0, 0)
            source = 1
        while source < n:
            ring += 1
            capacity = _ring_capacity(ring)
            copies = symmetry or 1
            distinct = max(1, capacity // copies)
            step = 360.0 / (distinct * copies)
            sector = 360.0 / copies
            for copy in range(copies):
                for offset in range(distinct):
                    flipped = mirror and copy % 2 == 1
                    if flipped:
                        phi = (copy + 1) * sector - (offset + 0.5) * step
                    else:
                        phi = copy * sector + (offset + 0.5) * step
                    ring_slot(ring, phi, source + offset, flip=flipped)
            source += distinct

    ring_index = np.array([row[0] for row in rows], float)
    phis = np.array([row[1] for row in rows])
    radii = ring_index * pitch
    angles = np.where(ring_index == 0, 0.0, _rotation(phis, rotate))
    slots = make_slots(
        radii * np.cos(np.radians(phis)),
        radii * np.sin(np.radians(phis)),
        size=float(spacing),
        angle=list(angles),
        source=[row[2] for row in rows],
        flip=[row[3] for row in rows],
        group=[row[4] for row in rows],
    )
    slots["ring"] = ring_index.astype(int)
    return Placed(slots, default_assign=sequence())


def phyllotaxis(spacing=1.0, rotate="none"):
    """Layout: a sunflower disc, items placed at the golden angle, filled from the center outward.

    Args:
        spacing: Distance between neighbors, as a multiple of the largest item.
        rotate: As in `spiral`.
    """
    _check_rotate(rotate)
    return Layout("phyllotaxis", _phyllotaxis, {"spacing": spacing, "rotate": rotate})


def _phyllotaxis(arrangement, spacing, rotate):
    index = np.arange(len(arrangement))
    # Vogel's spiral packs one point per pi*c^2; this c keeps nearest neighbors about one
    # pitch apart.
    c = _pitch(arrangement, spacing) * 0.62
    radii = c * np.sqrt(index + 0.5)
    phis = index * GOLDEN_ANGLE
    slots = make_slots(
        radii * np.cos(np.radians(phis)),
        radii * np.sin(np.radians(phis)),
        size=float(spacing),
        angle=list(_rotation(phis % 360.0, rotate)),
    )
    return Placed(slots, default_assign=sequence())


# ---------------------------------------------------------------------------
# Positions from data
# ---------------------------------------------------------------------------


def scatter(snap=True, spacing=1.0, density=1.5, extent=None, backdrop=None):
    """Layout: each item at its own position, from `position`, `polar_position` or `project`.

    Args:
        snap: Snap items to a grid, each to its own cell, so none overlap; otherwise each sits
            exactly at its position.
        spacing: Cell pitch, as a multiple of the largest item.
        density: Grid cells per item when snapping; more keeps positions truer.
        extent: `(xmin, xmax, ymin, ymax)` in data units, e.g. a map's bounds; the data's own
            range if None.
        backdrop: An image, or a path to one, drawn under the items across `extent`.
    """
    if backdrop is not None and extent is None:
        raise ValueError("a backdrop needs extent= to say where in the data it lies")
    data = {}
    parameters = {
        "snap": bool(snap),
        "spacing": spacing,
        "density": density,
        "extent": None if extent is None else [float(value) for value in extent],
        "backdrop": None,
    }
    if backdrop is not None:
        data["image"] = read_image(backdrop)
        parameters["backdrop"] = array_digest(data["image"])
    return Layout("scatter", _scatter, parameters, data=data)


def _scatter(arrangement, snap, spacing, density, extent, backdrop, image=None):
    if arrangement.targets is None:
        raise ValueError("scatter() needs positions: arrange with position(), polar_position() or project()")
    n = len(arrangement)
    targets = np.asarray(arrangement.targets, float)
    if extent is None:
        low, high = targets.min(axis=0), targets.max(axis=0)
        extent = [low[0], high[0], low[1], high[1]]
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    pitch = _pitch(arrangement, spacing)
    aspect = float(np.clip(dx / dy, 0.1, 10.0))
    area = n * (density if snap else 1.0) * pitch**2
    height = math.sqrt(area / aspect)
    width = aspect * height
    placed_targets = np.column_stack(
        [(targets[:, 0] - xmin) / dx * width, (ymax - targets[:, 1]) / dy * height]
    )

    decorations = []
    if image is not None:
        decorations.append(
            {
                "kind": "image",
                "x": 0.0,
                "y": 0.0,
                "width": width,
                "height": height,
                "image_hash": backdrop,
                "image": image,
                "layer": "under",
            }
        )

    if not snap:
        slots = make_slots(
            placed_targets[:, 0], placed_targets[:, 1], size=float(spacing), item=list(range(n))
        )
        return Placed(slots, decorations, targets=placed_targets, default_assign=sequence())

    columns = max(1, int(round(width / pitch)))
    rows = max(1, int(round(height / pitch)))
    while columns * rows < n:
        columns, rows = columns + 1, rows + 1
    index = np.arange(columns * rows)
    xs = (index % columns + 0.5) * width / columns
    ys = (index // columns + 0.5) * height / rows
    slots = make_slots(xs, ys, size=float(spacing))
    return Placed(slots, decorations, targets=placed_targets, default_assign=match(on="position"))


def beeswarm(spacing=1.0, width=None, ticks=6, label=None):
    """Layout: items along one axis by their position, stacked outward from it where they would overlap.

    Args:
        spacing: Distance kept between items, as a multiple of the largest item.
        width: Length of the axis in layout units; about three per square root of the item
            count if None.
        ticks: About how many labelled ticks to draw on the axis; 0 for none.
        label: Axis title; the position column's name if None.
    """
    return Layout(
        "beeswarm", _beeswarm, {"spacing": spacing, "width": width, "ticks": int(ticks), "label": label}
    )


def _swarm_offset(x, placed, pitch):
    """Return the vertical offset nearest the axis at which an item at `x` clears every placed item."""
    blocked = []
    for other_x, other_y in placed:
        dx = abs(x - other_x)
        if dx < pitch:
            reach = math.sqrt(pitch**2 - dx**2)
            blocked.append((other_y - reach, other_y + reach))
    candidates = sorted(
        {0.0, *(edge for interval in blocked for edge in interval)}, key=lambda y: (abs(y), y)
    )
    for y in candidates:
        if all(not (low + 1e-9 < y < high - 1e-9) for low, high in blocked):
            return y
    return 0.0


def _beeswarm(arrangement, spacing, width, ticks, label):
    if arrangement.targets is None:
        raise ValueError("beeswarm() needs positions: arrange with position(x=...) first")
    n = len(arrangement)
    pitch = _pitch(arrangement, spacing)
    values = np.asarray(arrangement.targets, float)[:, 0]
    low, high = float(values.min()), float(values.max())
    span = (high - low) or 1.0
    width = float(width) if width else max(10.0, 3.0 * math.sqrt(n)) * pitch
    xs = (values - low) / span * width

    order = np.lexsort((np.arange(n), xs))
    placed, ys = [], np.zeros(n)
    for index in order:
        ys[index] = _swarm_offset(xs[index], placed, pitch)
        placed.append((xs[index], ys[index]))

    decorations = []
    axis_y = ys.max() + pitch
    if ticks:
        from matplotlib.ticker import MaxNLocator

        decorations.append(line([(0.0, axis_y), (width, axis_y)], pitch * 0.02))
        for value in MaxNLocator(ticks).tick_values(low, high):
            if low - 1e-9 <= value <= high + 1e-9:
                x = (value - low) / span * width
                decorations.append(line([(x, axis_y), (x, axis_y + pitch * 0.15)], pitch * 0.02))
                decorations.append(text(x, axis_y + pitch * 0.2, f"{value:g}", pitch * 0.25))
        title = label or (arrangement.target_columns or ("",))[0]
        if title:
            decorations.append(text(width / 2.0, axis_y + pitch * 0.6, title, pitch * 0.3))

    slots = make_slots(xs, ys, size=float(spacing), item=list(range(n)))
    return Placed(slots, decorations, default_assign=sequence())


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def stack(spacing=1.0, alpha=None):
    """Layout: every item in one slot, one over another, as a ghost of their variation.

    Use with `blend="mean"`, or a low `alpha`, and oriented cutouts.

    Args:
        spacing: Slot size, as a multiple of the largest item.
        alpha: Opacity of each item; 1 if None.
    """
    return Layout("stack", _stack, {"spacing": spacing, "alpha": alpha})


def _stack(arrangement, spacing, alpha):
    n = len(arrangement)
    slots = make_slots(
        np.zeros(n), np.zeros(n), size=float(spacing), alpha=1.0 if alpha is None else float(alpha)
    )
    return Placed(slots, default_assign=sequence())


def mirrored(columns=4, spacing=1.0):
    """Layout: a specimen plate, slots in mirror-image pairs about a vertical axis, filled from the center.

    The right-hand slot of each pair is flipped. With items sorted largest first, the largest sit
    at the center and each pair is two of a similar size.

    Args:
        columns: Slots per row; an odd count puts a column on the axis.
        spacing: Slot pitch, as a multiple of the largest item.
    """
    return Layout("mirrored", _mirrored, {"columns": int(columns), "spacing": spacing})


def _mirrored(arrangement, columns, spacing):
    n = len(arrangement)
    pitch = _pitch(arrangement, spacing)
    rows = math.ceil(n / columns)
    offsets = np.arange(columns) - (columns - 1) / 2.0
    candidates = []
    for row in range(rows):
        row_offset = row - (rows - 1) / 2.0
        for offset in offsets:
            candidates.append((abs(row_offset), abs(offset), row_offset, offset > 0, offset))
    candidates.sort()
    chosen = candidates[:n]
    slots = make_slots(
        [offset * pitch for *_, offset in chosen],
        [row_offset * pitch for _, _, row_offset, _, _ in chosen],
        size=float(spacing),
        flip=[bool(right) for _, _, _, right, _ in chosen],
    )
    return Placed(slots, default_assign=sequence())


def exploded(offsets, spacing=1.0):
    """Layout: one occurrence's parts at fixed offsets from each other, as an exploded diagram.

    Meant inside `nested` with `group_by("occurrence_id")`, one occurrence per group.

    Args:
        offsets: `{part: (dx, dy)}` in layout units; a part not listed sits at the center.
        spacing: Slot size, as a multiple of the largest item.
    """
    offsets = {str(part): [float(dx), float(dy)] for part, (dx, dy) in offsets.items()}
    return Layout("exploded", _exploded, {"offsets": offsets, "spacing": spacing})


def _exploded(arrangement, offsets, spacing):
    points = [offsets.get(part, [0.0, 0.0]) for _, part in arrangement.keys]
    slots = make_slots(
        [point[0] for point in points],
        [point[1] for point in points],
        size=float(spacing),
        item=list(range(len(arrangement))),
    )
    return Placed(slots, default_assign=sequence())


def _top_groups(arrangement):
    """Return `[(label, positions)]` for each outermost group, in order of first appearance."""
    if arrangement.groups is None:
        raise ValueError("this layout needs groups: arrange with group_by() first")
    groups = {}
    for position, labels in enumerate(arrangement.groups):
        groups.setdefault(labels[0], []).append(position)
    return list(groups.items())


def _inner(arrangement, positions):
    """Return the arrangement of some items with their outermost group label removed."""
    inner = arrangement.take(positions)
    rest = [labels[1:] for labels in inner.groups]
    inner.groups = rest if all(rest) else None
    return inner


def resolve_items(placed, arrangement, assign=None):
    """Return the placed slots with `item` filled in, by the slots' own items or an assignment."""
    assignment = assign or placed.default_assign or sequence()
    return assignment(arrangement, placed)


def _extent(placements, round_extent):
    """Return the center and diameter of what a set of placements covers."""
    half = placements["size"].to_numpy() / 2.0
    xs, ys = placements["x"].to_numpy(), placements["y"].to_numpy()
    if round_extent:
        cx, cy = xs.mean(), ys.mean()
        return cx, cy, float(2.0 * (np.hypot(xs - cx, ys - cy) + half).max())
    left, right = (xs - half).min(), (xs + half).max()
    top, bottom = (ys - half).min(), (ys + half).max()
    return (left + right) / 2.0, (top + bottom) / 2.0, float(max(right - left, bottom - top))


def _moved(decoration, cx, cy, factor, x, y):
    """Return a decoration scaled about `(cx, cy)` by `factor` and moved to `(x, y)`."""
    decoration = dict(decoration)

    def point(px, py):
        return x + (px - cx) * factor, y + (py - cy) * factor

    if "points" in decoration:
        decoration["points"] = [list(point(px, py)) for px, py in decoration["points"]]
    if "x" in decoration:
        decoration["x"], decoration["y"] = point(decoration["x"], decoration["y"])
    for key in ("height", "width"):
        if key in decoration:
            decoration[key] = decoration[key] * factor
    return decoration


def nested(outer, inner, inner_assign=None, labels=False, padding=0.15):
    """Layout: each group laid out by `inner`, and the groups laid out by `outer`.

    Groups are the outermost `group_by` labels; `inner` sees the rest of the hierarchy, so
    layouts nest as deep as it goes. `pack_circles` as `outer` sizes each group by its contents.

    Args:
        outer: The layout of the groups.
        inner: The layout of the items in each group.
        inner_assign: The assignment within each group; `inner`'s default if None.
        labels: Write each group's label under it.
        padding: Space kept around each group, as a fraction of its size.
    """
    return Layout(
        "nested",
        _nested,
        {
            "outer": outer.spec(),
            "inner": inner.spec(),
            "inner_assign": None if inner_assign is None else inner_assign.spec(),
            "labels": bool(labels),
            "padding": padding,
        },
        data={"outer_layout": outer, "inner_layout": inner, "assignment": inner_assign},
    )


def _nested(arrangement, outer, inner, inner_assign, labels, padding, outer_layout, inner_layout, assignment):
    from .arrange import Arrangement

    round_extent = outer_layout.name == "pack_circles"
    groups, blocks = _top_groups(arrangement), []
    for label, positions in groups:
        sub = _inner(arrangement, positions)
        placed = inner_layout.place(sub)
        placements = resolve_items(placed, sub, assignment)
        cx, cy, diameter = _extent(placements, round_extent)
        blocks.append((label, positions, placed, placements, cx, cy, max(diameter, 1e-6) * (1 + padding)))

    diameters = np.array([block[-1] for block in blocks])
    group_arrangement = Arrangement(
        rows=np.arange(len(blocks)),
        keys=[(str(block[0]), "") for block in blocks],
        scale=diameters / diameters.max(),
    )
    outer_slots = resolve_items(outer_layout.place(group_arrangement), group_arrangement)

    frames, decorations = [], []
    for outer_row in outer_slots.itertuples(index=False):
        label, positions, placed, placements, cx, cy, diameter = blocks[outer_row.item]
        # The group drawn at its outer slot's size (already scaled by the group's extent).
        factor = outer_row.size / diameter
        frame = placements.copy()
        frame["x"] = outer_row.x + (frame["x"] - cx) * factor
        frame["y"] = outer_row.y + (frame["y"] - cy) * factor
        # Back to a slot size: the outer assignment scales each item again.
        items = [positions[item] for item in frame["item"]]
        frame["size"] = frame["size"] * factor / arrangement.scales()[items]
        frame["clip_size"] = frame["clip_size"] * factor
        frame["item"] = items
        frame["group"] = label
        frames.append(frame)
        decorations += [_moved(d, cx, cy, factor, outer_row.x, outer_row.y) for d in placed.decorations]
        if labels:
            decorations.append(
                text(outer_row.x, outer_row.y + outer_row.size / 2.0, label, 0.1 * outer_row.size)
            )

    slots = pd.concat(frames, ignore_index=True)
    slots["source"] = np.arange(len(slots))
    slots = slots.drop(columns=["slot"], errors="ignore")
    return Placed(slots, decorations, default_assign=sequence())


def pack_circles(padding=0.05):
    """Layout: items as circles, sized by their scale, packed tightly around the center, largest first.

    As `nested`'s `outer`, each circle is a group sized by its contents.

    Args:
        padding: Gap kept between circles, as a fraction of the largest diameter.
    """
    return Layout("pack_circles", _pack_circles, {"padding": padding})


def _pack_circles(arrangement, padding):
    n = len(arrangement)
    if n > MAX_CIRCLES:
        raise ValueError(f"pack_circles packs at most {MAX_CIRCLES} circles, got {n} -- pack groups instead")
    radii = arrangement.scales() / 2.0
    gap = padding * 2.0 * radii.max()
    order = sorted(range(n), key=lambda index: (-radii[index], index))
    centers = np.zeros((n, 2))
    placed = []
    for index in order:
        r = radii[index]
        if not placed:
            centers[index] = 0.0
        elif len(placed) == 1:
            other = placed[0]
            centers[index] = centers[other] + [radii[other] + r + gap, 0.0]
        else:
            centers[index] = _best_tangent(centers[placed], radii[placed] + gap / 2.0, r + gap / 2.0)
        placed.append(index)
    slots = make_slots(centers[:, 0], centers[:, 1], size=1.0, item=list(range(n)))
    return Placed(slots, default_assign=sequence())


def _best_tangent(centers, radii, r):
    """Return the position nearest the origin where a circle of radius `r` touches two placed ones and overlaps none."""
    candidates = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            a, b = radii[i] + r, radii[j] + r
            delta = centers[j] - centers[i]
            d = float(np.hypot(*delta))
            if d == 0 or d > a + b or d < abs(a - b):
                continue
            along = (a**2 - b**2 + d**2) / (2 * d)
            across = math.sqrt(max(a**2 - along**2, 0.0))
            base = centers[i] + along * delta / d
            normal = np.array([-delta[1], delta[0]]) / d
            candidates += [base + across * normal, base - across * normal]
    if not candidates:
        far = float(np.hypot(centers[:, 0], centers[:, 1]).max() + radii.max() + r)
        return np.array([far, 0.0])
    candidates = np.array(candidates)
    distances = np.hypot(
        candidates[:, None, 0] - centers[None, :, 0], candidates[:, None, 1] - centers[None, :, 1]
    )
    clear = (distances >= radii[None, :] + r - 1e-9).all(axis=1)
    candidates = candidates[clear]
    if not len(candidates):
        far = float(np.hypot(centers[:, 0], centers[:, 1]).max() + radii.max() + r)
        return np.array([far, 0.0])
    norms = np.hypot(candidates[:, 0], candidates[:, 1])
    best = np.lexsort((candidates[:, 1], candidates[:, 0], np.round(norms, 9)))[0]
    return candidates[best]


def dendrogram(radial=True, spacing=1.0, level=None, labels=True):
    """Layout: items as the leaves of the tree their `group_by` hierarchy draws, with its edges.

    Args:
        radial: Leaves on a circle around the root; otherwise along a line under it.
        spacing: Distance between leaves, as a multiple of the largest item.
        level: Distance between tree levels in layout units; one pitch if None.
        labels: Label the clades of the innermost group level.
    """
    return Layout(
        "dendrogram",
        _dendrogram,
        {"radial": bool(radial), "spacing": spacing, "level": level, "labels": bool(labels)},
    )


def _dendrogram(arrangement, radial, spacing, level, labels):
    if arrangement.groups is None:
        raise ValueError("dendrogram() needs a hierarchy: arrange with group_by([...]) first")
    n = len(arrangement)
    pitch = _pitch(arrangement, spacing)
    level = float(level) if level else pitch
    depth = max(len(groups) for groups in arrangement.groups)
    paths = [tuple(groups) + ("",) * (depth - len(groups)) for groups in arrangement.groups]
    leaves = sorted(range(n), key=lambda index: (paths[index], index))
    leaf_at = {index: rank for rank, index in enumerate(leaves)}

    # Each clade is a path prefix; its position is the mean of its leaves'.
    clades = {}
    for index in leaves:
        for d in range(depth + 1):
            clades.setdefault(paths[index][:d], []).append(leaf_at[index])

    if radial:
        radius = max(n * pitch / (2.0 * math.pi), (depth + 1) * level)
        step = radius / (depth + 1)

        def point(t, d):
            angle = 2.0 * math.pi * t / n
            r = step * d if d <= depth else radius
            return r * math.cos(angle), r * math.sin(angle)

    else:

        def point(t, d):
            return t * pitch, d * level

    def where(prefix):
        d = len(prefix)
        return float(np.mean(clades[prefix])), d

    decorations = []
    width = pitch * 0.02
    for prefix, members in clades.items():
        if len(prefix) == depth:
            children = [(t, depth + 1) for t in members]
        else:
            children = [
                where(child)
                for child in clades
                if len(child) == len(prefix) + 1 and child[: len(prefix)] == prefix
            ]
        t0, d0 = where(prefix)
        ts = [t for t, _ in children]
        # The bar joining the children at this clade's level, then a drop to each.
        bar = (
            np.linspace(min(ts), max(ts), max(2, int(abs(max(ts) - min(ts)) * 4) + 2))
            if radial
            else [min(ts), max(ts)]
        )
        decorations.append(line([point(t, d0) for t in bar], width))
        for t, d in children:
            decorations.append(line([point(t, d0), point(t, d)], width))

    if labels and depth:
        for prefix, members in clades.items():
            if len(prefix) == depth and prefix[-1]:
                t = float(np.mean(members))
                x, y = (
                    point(t, depth + 1.6)
                    if not radial
                    else (
                        (radius + 1.2 * pitch) * math.cos(2 * math.pi * t / n),
                        (radius + 1.2 * pitch) * math.sin(2 * math.pi * t / n),
                    )
                )
                decorations.append(text(x, y, prefix[-1], pitch * 0.25, valign="center"))

    xs, ys, angles = [], [], []
    for index in range(n):
        x, y = point(leaf_at[index], depth + 1)
        xs.append(x)
        ys.append(y)
        angles.append(_rotation(math.degrees(2 * math.pi * leaf_at[index] / n), "radial") if radial else 0.0)
    slots = make_slots(xs, ys, size=float(spacing), angle=[float(a) for a in angles], item=list(range(n)))
    return Placed(slots, decorations, default_assign=sequence())
