"""Decorations: what a collage draws besides specimens -- labels, swatches, a scale bar, a caption."""

import numpy as np
import pandas as pd

from .base import Step
from .layouts import DECORATION_COLOR, text

RGB_KEYS = ("r", "g", "b")


class Decoration(Step):
    """A step from the placed items to decorations in layout units.

    Args:
        name: As in `Step`.
        function: Called as `function(placements, table, arrangement, **parameters)`, returning
            a list of decorations.
        parameters: As in `Step`.
        columns: As in `Step`.
    """

    kind = "decoration"

    def __init__(self, name, function, parameters=None, columns=()):
        super().__init__(name, parameters, columns)
        self.function = function

    def resolve(self, placements, table, arrangement):
        """Return this decoration's drawing for these placements."""
        return self.function(placements, table, arrangement, **self.parameters)


def _value(table, arrangement, column, item):
    value = table[column].to_numpy()[arrangement.rows[item]]
    return None if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)) else value


def _bounds(placements):
    """Return `(left, top, right, bottom)` of the placed items' slots."""
    half = placements["size"].to_numpy() / 2.0
    xs, ys = placements["x"].to_numpy(), placements["y"].to_numpy()
    return (xs - half).min(), (ys - half).min(), (xs + half).max(), (ys + half).max()


def label(column=None, height=0.12, color=DECORATION_COLOR, digits=3):
    """Decoration: text under each item, from a column, or numbers in placing order.

    Args:
        column: The column; None numbers the items 1, 2, 3, ...
        height: Text height, as a fraction of the item's size.
        color: BGR text color.
        digits: Significant digits for a number.
    """
    return Decoration(
        "label",
        _label,
        {"column": column, "height": height, "color": list(color), "digits": int(digits)},
        columns=[] if column is None else [column],
    )


def _label(placements, table, arrangement, column, height, color, digits):
    out = []
    for number, row in enumerate(placements.itertuples(index=False), start=1):
        if column is None:
            content = str(number)
        else:
            value = _value(table, arrangement, column, row.item)
            if value is None:
                continue
            content = f"{value:.{digits}g}" if isinstance(value, (float, np.floating)) else str(value)
        out.append(text(row.x, row.y + row.size / 2.0, content, height * row.size, color))
    return out


def swatch(colors, size=0.18):
    """Decoration: a column of color squares beside each item, e.g. its palette.

    Args:
        colors: Column prefixes, each with `__r`, `__g` and `__b` columns on 0-1 scales, as
            `mean_color()` stores them, e.g. `"colors__organism__mean_color"`.
        size: Square side, as a fraction of the item's size.
    """
    colors = [colors] if isinstance(colors, str) else list(colors)
    return Decoration(
        "swatch",
        _swatch,
        {"colors": colors, "size": size},
        columns=[f"{prefix}__{key}" for prefix in colors for key in RGB_KEYS],
    )


def _swatch(placements, table, arrangement, colors, size):
    out = []
    for row in placements.itertuples(index=False):
        side = size * row.size
        left = row.x + row.size / 2.0 + side * 0.25
        top = row.y - side * len(colors) / 2.0
        for index, prefix in enumerate(colors):
            rgb = [_value(table, arrangement, f"{prefix}__{key}", row.item) for key in RGB_KEYS]
            if any(value is None for value in rgb):
                continue
            bgr = [int(round(255 * float(np.clip(value, 0, 1)))) for value in reversed(rgb)]
            out.append(
                {
                    "kind": "rect",
                    "x": float(left),
                    "y": float(top + index * side),
                    "width": float(side),
                    "height": float(side),
                    "color": bgr,
                    "layer": "over",
                }
            )
    return out


def scale_bar(length, unit=None, color=DECORATION_COLOR):
    """Decoration: a bar under the collage showing how long `length` of the `scale_by` column is.

    Needs `scale_by` among the arrangement steps.

    Args:
        length: The bar's length, in the `scale_by` column's unit.
        unit: Unit text; the column's recorded unit if None.
        color: BGR color.
    """
    return Decoration("scale_bar", _scale_bar, {"length": float(length), "unit": unit, "color": list(color)})


def _scale_bar(placements, table, arrangement, length, unit, color):
    reference = arrangement.scale_reference
    if reference is None:
        raise ValueError("scale_bar() needs item sizes from scale_by() among the arrange= steps")
    scales = arrangement.scales()[placements["item"].to_numpy()]
    units_per_value = float(np.median(placements["size"].to_numpy() / scales)) / reference["value"]
    left, _, _, bottom = _bounds(placements)
    typical = float(np.median(placements["size"]))
    y = bottom + 0.3 * typical
    bar = {
        "kind": "rect",
        "x": float(left),
        "y": float(y),
        "width": length * units_per_value,
        "height": 0.04 * typical,
        "color": list(color),
        "layer": "over",
    }
    unit = unit or reference.get("unit") or ""
    caption = text(
        left + length * units_per_value / 2.0,
        y + 0.08 * typical,
        f"{length:g} {unit}".strip(),
        0.15 * typical,
        color,
    )
    return [bar, caption]


def caption(content, height=0.4, color=DECORATION_COLOR):
    """Decoration: a line of text centered above the collage.

    Args:
        content: The text.
        height: Text height, as a multiple of the typical item's size.
        color: BGR color.
    """
    return Decoration("caption", _caption, {"content": str(content), "height": height, "color": list(color)})


def _caption(placements, table, arrangement, content, height, color):
    left, top, right, _ = _bounds(placements)
    typical = float(np.median(placements["size"]))
    return [text((left + right) / 2.0, top - (height + 0.3) * typical, content, height * typical, color)]
