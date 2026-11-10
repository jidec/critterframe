"""Arrange: stored values into an item order, groups, sizes, positions and features for a layout."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ...records.occurrences import ID_COL
from .base import Step

PART_COL = "part"
MISSING_GROUP = "unknown"


@dataclass
class Arrangement:
    """The items of a collage in order, with what each layout and assignment may read about them.

    Attributes:
        rows: Each item's row in the collage table.
        keys: Each item's `(occurrence_id, part)`.
        groups: Each item's group labels, outermost first, or None.
        scale: Each item's size factor, or None for all 1.
        targets: Each item's `(x, y)` position in data units, or None.
        values: Each item's feature vector for matching, or None.
        scale_reference: `{"column", "value", "unit"}` behind `scale`, or None.
        target_columns: The columns behind `targets`, or None.
    """

    rows: np.ndarray
    keys: list
    groups: list = None
    scale: np.ndarray = None
    targets: np.ndarray = None
    values: np.ndarray = None
    scale_reference: dict = None
    target_columns: tuple = None

    @classmethod
    def of(cls, table):
        """Return the table's items in its own order, which is sorted by key."""
        keys = list(zip(table[ID_COL].astype(str), table[PART_COL].astype(str)))
        return cls(rows=np.arange(len(table)), keys=keys)

    def __len__(self):
        return len(self.rows)

    def scales(self):
        """Return each item's size factor, 1 where none was set."""
        return np.ones(len(self)) if self.scale is None else np.asarray(self.scale, float)

    def take(self, positions):
        """Return the arrangement of these items, in this order."""
        positions = np.asarray(positions, dtype=int)

        def pick(array):
            return None if array is None else np.asarray(array)[positions]

        return replace(
            self,
            rows=self.rows[positions],
            keys=[self.keys[position] for position in positions],
            groups=None if self.groups is None else [self.groups[position] for position in positions],
            scale=pick(self.scale),
            targets=pick(self.targets),
            values=pick(self.values),
        )


class Arranger(Step):
    """A step from `(arrangement, table)` to a new arrangement.

    Args:
        name: As in `Step`.
        function: Called as `function(arrangement, table, **parameters)`.
        parameters: As in `Step`.
        columns: As in `Step`.
    """

    kind = "arrange"

    def __init__(self, name, function, parameters=None, columns=()):
        super().__init__(name, parameters, columns)
        self.function = function

    def __call__(self, arrangement, table):
        """Apply the step."""
        return self.function(arrangement, table, **self.parameters)


def _columns(columns):
    """Return one column name or several as a list."""
    return [columns] if isinstance(columns, str) else list(columns)


def column_values(table, arrangement, column):
    """Return a column's value for each item of the arrangement, in its order."""
    return table[column].to_numpy()[arrangement.rows]


def feature_matrix(table, arrangement, columns):
    """Return an `(items, features)` float array from one vector-valued column or several scalar ones."""
    columns = _columns(columns)
    if len(columns) == 1:
        values = column_values(table, arrangement, columns[0])
        if len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
            return np.vstack([np.asarray(value, float) for value in values])
        return np.asarray(values, float).reshape(-1, 1)
    return np.column_stack([np.asarray(column_values(table, arrangement, c), float) for c in columns])


def sort_by(column, descending=False, circular=False, start=0.0, period=360.0):
    """Arrange: order items by a column, ties kept in their earlier order.

    Args:
        column: The column, e.g. `"colors__organism__mean_color__h"`.
        descending: Largest first.
        circular: The value wraps, as a hue does; the order starts at `start`.
        start: Where a circular order begins.
        period: What a circular value wraps at.
    """
    return Arranger(
        "sort_by",
        _sort_by,
        {"column": column, "descending": descending, "circular": circular, "start": start, "period": period},
        columns=[column],
    )


def _sort_by(arrangement, table, column, descending, circular, start, period):
    key = np.asarray(column_values(table, arrangement, column), float)
    if circular:
        key = (key - start) % period
    order = np.argsort(-key if descending else key, kind="stable")
    return arrangement.take(order)


def group_by(columns, order="name"):
    """Arrange: group items by one column, or by several as a hierarchy, outermost first.

    Items are reordered so each group is contiguous, keeping their earlier order inside it.

    Args:
        columns: The column, or the columns from outermost to innermost, e.g.
            `["family", "genus", "species"]`.
        order: `"name"` sorts groups by label, `"first"` keeps the order their first items had,
            `"size"` puts the largest group first.
    """
    if order not in ("name", "first", "size"):
        raise ValueError(f"order must be 'name', 'first' or 'size', got {order!r}")
    return Arranger(
        "group_by", _group_by, {"columns": _columns(columns), "order": order}, columns=_columns(columns)
    )


def _label(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return MISSING_GROUP
    return str(value)


def _group_by(arrangement, table, columns, order):
    labels = [
        tuple(_label(value) for value in values)
        for values in zip(*(column_values(table, arrangement, column) for column in columns))
    ]
    if order == "name":
        rank = {label: index for index, label in enumerate(sorted(set(labels)))}
    elif order == "first":
        rank = {}
        for label in labels:
            rank.setdefault(label, len(rank))
    else:
        counts = pd.Series(labels).value_counts()
        ranked = sorted(set(labels), key=lambda label: (-counts[label], label))
        rank = {label: index for index, label in enumerate(ranked)}
    positions = np.argsort([rank[label] for label in labels], kind="stable")
    arranged = arrangement.take(positions)
    arranged.groups = [labels[position] for position in positions]
    return arranged


def scale_by(column, reference=None):
    """Arrange: size each item in proportion to a column, its longest side standing for the value.

    Args:
        column: The column, e.g. a body length.
        reference: The value drawn at size 1; the median if None.
    """
    return Arranger("scale_by", _scale_by, {"column": column, "reference": reference}, columns=[column])


def _scale_by(arrangement, table, column, reference):
    values = np.asarray(column_values(table, arrangement, column), float)
    if reference is None:
        reference = float(np.median(values)) if len(values) else 1.0
    if not reference > 0:
        raise ValueError(f"scale_by('{column}') needs a positive reference value, got {reference}")
    arranged = replace(arrangement, scale=np.clip(values / reference, 1e-3, None))
    arranged.scale_reference = {"column": column, "value": float(reference), "unit": None}
    return arranged


def position(x, y=None):
    """Arrange: place each item at a position read from columns, e.g. longitude and latitude.

    Args:
        x: The horizontal column.
        y: The vertical column, increasing upward; None puts every item at 0.
    """
    columns = [x] if y is None else [x, y]
    return Arranger("position", _position, {"x": x, "y": y}, columns=columns)


def _position(arrangement, table, x, y):
    xs = np.asarray(column_values(table, arrangement, x), float)
    ys = np.zeros(len(xs)) if y is None else np.asarray(column_values(table, arrangement, y), float)
    return replace(arrangement, targets=np.column_stack([xs, ys]), target_columns=(x, y))


def polar_position(angle, radius=None):
    """Arrange: place each item by an angle in degrees and a radius, e.g. hue and chroma.

    Args:
        angle: The angle column, counterclockwise from the right.
        radius: The radius column; None puts every item on the unit circle.
    """
    columns = [angle] if radius is None else [angle, radius]
    return Arranger("polar_position", _polar_position, {"angle": angle, "radius": radius}, columns=columns)


def _polar_position(arrangement, table, angle, radius):
    radians = np.radians(np.asarray(column_values(table, arrangement, angle), float))
    radii = (
        np.ones(len(radians))
        if radius is None
        else np.asarray(column_values(table, arrangement, radius), float)
    )
    targets = np.column_stack([radii * np.cos(radians), radii * np.sin(radians)])
    return replace(arrangement, targets=targets, target_columns=(angle, radius))


def project(columns, standardize=None):
    """Arrange: place each item by the first two principal components of its features.

    Args:
        columns: One vector-valued column, such as an embedding, or several scalar ones.
        standardize: Scale each feature to unit variance first; by default only for several
            scalar columns, whose units differ.
    """
    columns = _columns(columns)
    if standardize is None:
        standardize = len(columns) > 1
    return Arranger("project", _project, {"columns": columns, "standardize": standardize}, columns=columns)


def principal_components(features, standardize=False, n=2):
    """Return the first `n` principal component scores, each signed so its largest loading is positive."""
    features = np.asarray(features, float)
    centered = features - features.mean(axis=0)
    if standardize:
        spread = centered.std(axis=0)
        centered = centered / np.where(spread > 0, spread, 1.0)
    if len(centered) < 2:
        return np.zeros((len(centered), n))
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:n]
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    scores = centered @ (components * signs[:, None]).T
    if scores.shape[1] < n:
        scores = np.column_stack([scores, np.zeros((len(scores), n - scores.shape[1]))])
    return scores


def _project(arrangement, table, columns, standardize):
    scores = principal_components(feature_matrix(table, arrangement, columns), standardize)
    # Data coordinates increase upward, so the first component reads left to right and
    # the second bottom to top.
    return replace(arrangement, targets=scores, target_columns=("pc1", "pc2"))


def match_on(columns):
    """Arrange: give each item the features an assignment matches slots against, e.g. a mean color.

    Args:
        columns: One vector-valued column, or several scalar ones in the order the slots'
            values have, e.g. the `r`, `g` and `b` of `mean_color`.
    """
    columns = _columns(columns)
    return Arranger("match_on", _match_on, {"columns": columns}, columns=columns)


def _match_on(arrangement, table, columns):
    return replace(arrangement, values=feature_matrix(table, arrangement, columns))


def path(columns, start, end, neighbors=8):
    """Arrange: keep only the items on the shortest path between two occurrences through their features.

    The path runs through a graph joining each item to its nearest neighbors, weighted by squared
    distance, so it takes many small steps over one large one. Items off the path are left out.

    Args:
        columns: As in `project`.
        start: The occurrence the path starts at.
        end: The occurrence it ends at.
        neighbors: How many nearest neighbors each item is joined to.
    """
    columns = _columns(columns)
    return Arranger(
        "path",
        _path,
        {"columns": columns, "start": str(start), "end": str(end), "neighbors": int(neighbors)},
        columns=columns,
    )


def _path(arrangement, table, columns, start, end, neighbors):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    ids = [key[0] for key in arrangement.keys]
    for name, occurrence_id in (("start", start), ("end", end)):
        if occurrence_id not in ids:
            raise ValueError(f"path {name} {occurrence_id!r} is not among the collage's items")
    source, target = ids.index(start), ids.index(end)

    features = feature_matrix(table, arrangement, columns)
    distances = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    k = min(neighbors, len(features) - 1)
    weights = np.zeros_like(distances)
    for index in range(len(features)):
        nearest = [other for other in np.argsort(distances[index], kind="stable") if other != index][:k]
        # A tiny floor keeps two identical feature vectors joined; a zero weight is no edge.
        weights[index, nearest] = weights[nearest, index] = np.maximum(distances[index, nearest] ** 2, 1e-12)
    graph = csr_matrix(weights)
    _, predecessors = dijkstra(graph, indices=source, return_predecessors=True)
    if source != target and predecessors[target] < 0:
        raise ValueError(
            f"no path from {start!r} to {end!r} through {neighbors} nearest neighbors -- raise neighbors="
        )
    chain = [target]
    while chain[-1] != source:
        chain.append(predecessors[chain[-1]])
    return arrangement.take(chain[::-1])
