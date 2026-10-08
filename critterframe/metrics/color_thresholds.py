"""
Colour thresholds: named cutoffs on colour channels, across colour spaces, and the fraction of an organism past them.

`inductive_color_thresholds` fits the same kind of threshold from the data instead of taking it as given.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..colorspaces import convert, get_space, in_arc
from ..recipes import Metric
from ..visualization.panels import annotate, side_by_side
from .derived import DerivedMetric, derived
from .pixels import masked_pixels

# Hue arcs in degrees, half-open [start, end). Red wraps through 0, which
# `colorspaces.in_arc` handles -- the single most common way a hue filter quietly
# loses half its pixels. The gaps between arcs (pinks, 322-340) belong to no band.
HUE_BANDS = {
    "red": (340, 22),
    "orange": (22, 42),
    "yellow": (42, 72),
    "green": (72, 172),
    "blue": (172, 262),
    "purple": (262, 322),
}

# Floors below which a pixel has no meaningful hue: too grey to be any colour,
# or too dark to tell. Both on 0-1 scales.
MIN_SATURATION = 0.25
MIN_VALUE = 0.15

# Lightness below which a pixel counts as black, on a 0-1 scale.
BLACK_THRESHOLD = 0.20

# The key `threshold_fractions(unmatched=True)` reports pixels matching no threshold under.
UNMATCHED = "unmatched"

# The operation name color_presence recognizes a threshold_fractions metric by.
THRESHOLD_FRACTIONS = "threshold_fractions"

# How bright an organism pixel a threshold did NOT match is drawn in its panel, as a fraction of its grey level.
_DIMMED = 0.35


@dataclass(frozen=True)
class ColorThreshold:
    """
    A named set of cutoffs on colour channels; a pixel matches when it satisfies every one.

    Bounds are half-open `[low, high)` in the canonical units of `colorspaces.convert`, with `None` for an open end.
    A circular channel (hue) takes an arc in degrees, which wraps when `low > high`, so it needs both ends.

    - `name` -- what the matched fraction is stored under; no `"__"`, which export column names are joined with.
    - `conditions` -- `{space: {channel: (low, high)}}`, e.g. `{"hsv": {"h": (42, 72), "s": (0.25, None)}}`.
    """

    name: str
    conditions: tuple

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a colour threshold needs a non-empty string name, got {self.name!r}")
        if "__" in self.name:
            raise ValueError(f"colour threshold name {self.name!r} contains '__', which export column names use")

        rows = self.conditions
        if isinstance(rows, dict):
            rows = [(space, channel, *bounds)
                    for space, channels in rows.items() for channel, bounds in channels.items()]
        rows = [_checked_condition(self.name, *row) for row in rows]
        if not rows:
            raise ValueError(f"colour threshold {self.name!r} has no conditions")

        keys = [(space, channel) for space, channel, _, _ in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"colour threshold {self.name!r} gives one channel two conditions")
        object.__setattr__(self, "conditions", tuple(sorted(rows, key=lambda row: row[:2])))

    @property
    def spaces(self):
        """The colour spaces this threshold reads, in sorted order."""
        return tuple(sorted({space for space, _, _, _ in self.conditions}))

    def spec(self):
        """The JSON description of this threshold, which is what reaches a recipe hash."""
        conditions = {}
        for space, channel, low, high in self.conditions:
            conditions.setdefault(space, {})[channel] = [low, high]
        return {"name": self.name, "conditions": conditions}

    @classmethod
    def from_spec(cls, spec):
        """
        Rebuild a threshold from its `spec()`.

        - `spec` -- `{"name": ..., "conditions": {space: {channel: [low, high]}}}`.

        Returns a `ColorThreshold`.
        """
        return cls(spec["name"], spec["conditions"])

    def matches(self, converted):
        """
        Which pixels satisfy every condition.

        - `converted` -- `{space: values}` from `colorspaces.convert`, holding at least every space in `spaces`.

        Returns a boolean array shaped like one space's values minus its channel axis.
        """
        matched = None
        for space, channel, low, high in self.conditions:
            values = converted[space][..., get_space(space).index(channel)]
            if channel in get_space(space).circular:
                inside = in_arc(values, low, high)
            else:
                inside = np.ones(values.shape, dtype=bool)
                if low is not None:
                    inside &= values >= low
                if high is not None:
                    inside &= values < high
            matched = inside if matched is None else matched & inside
        return matched


def color_threshold(name, **conditions):
    """
    Build a `ColorThreshold` from `<space>_<channel>=(low, high)` keywords.

    e.g. `color_threshold("yellow", hsv_h=(42, 72), hsv_s=(0.25, None), lab_l=(20, None))`.

    - `name` -- what the matched fraction is stored under.
    - `conditions` -- one keyword per channel cutoff, `None` for an open end.

    Returns a `ColorThreshold`.
    """
    nested = {}
    for key, bounds in conditions.items():
        space, separator, channel = key.partition("_")
        if not separator or not channel:
            raise ValueError(f"condition {key!r} should be named <space>_<channel>, e.g. hsv_h")
        nested.setdefault(space, {})[channel] = bounds
    return ColorThreshold(name, nested)


def hue_thresholds(min_saturation=MIN_SATURATION, min_value=MIN_VALUE):
    """
    The six named hue bands of `HUE_BANDS` as colour thresholds, each floored on saturation and value.

    - `min_saturation` -- pixels greyer than this match no hue, rather than a hue they don't really have.
    - `min_value` -- pixels darker than this match no hue; hue is meaningless in shadow.

    Returns a list of `ColorThreshold`, in `HUE_BANDS` order.
    """
    return [_hue_threshold(hue, min_saturation, min_value) for hue in HUE_BANDS]


def threshold_fractions(thresholds, unmatched=False, name=None, unit="fraction"):
    """
    Metric: the fraction of masked pixels matching each colour threshold, as `{threshold name: fraction}`.

    Thresholds are scored independently, so they may overlap and the fractions need not sum to 1.

    - `thresholds` -- `ColorThreshold`s, e.g. from `color_threshold()` or `hue_thresholds()`.
    - `unmatched` -- also report `"unmatched"`, the fraction matching none of them.
    """
    thresholds = list(thresholds)
    names = [threshold.name for threshold in thresholds]
    if not thresholds:
        raise ValueError("threshold_fractions needs at least one colour threshold")
    if len(set(names)) != len(names):
        raise ValueError(f"colour threshold names must be unique, got {names}")
    if unmatched and UNMATCHED in names:
        raise ValueError(f"a threshold named {UNMATCHED!r} collides with unmatched=True")

    return Metric(THRESHOLD_FRACTIONS, _threshold_fractions,
                  {"thresholds": [threshold.spec() for threshold in thresholds],
                   "unmatched": bool(unmatched)},
                  version="1", unit=unit, metric_name=name)


def color_presence(fractions=None, from_run=None, min_fraction=0.10, n_ranked_colors=2,
                   name="color_presence", unit=None):
    """
    Metric: which colours of a `threshold_fractions` result are present, how many, and the largest.

    Derived from the stored fractions: list it after a `threshold_fractions`
    in the same run, or pass that operation and `from_run`. Returns
    `{<colour>_present, n_colors_present, ranked_color_1 ... ranked_color_<n>}`;
    `"unmatched"` never counts, a colour with no stored fraction is absent,
    and ties keep the bins' order.

    - `fractions` -- which `threshold_fractions` to read: None for the one
      listed before this in the same run, its metric name where there are
      several, or the operation itself, configured as measured.
    - `from_run` -- the run that stored it, which needs the operation; None
      for an earlier metric in this run.
    - `min_fraction` -- the share of the organism a colour must cover to be present.
    - `n_ranked_colors` -- how many `ranked_color_<i>` to report, largest first; None past those present.
    """
    if not 0 < min_fraction <= 1:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction!r}")
    if isinstance(n_ranked_colors, bool) or not isinstance(n_ranked_colors, int) or n_ranked_colors < 0:
        raise ValueError(f"n_ranked_colors must be a non-negative integer, got {n_ranked_colors!r}")

    if fractions is None or isinstance(fractions, str):
        if from_run is not None:
            raise ValueError(
                f"color_presence needs the threshold_fractions(...) operation run {from_run!r} "
                "measured with -- only a metric of this same run can be found by name")
        return _UnboundColorPresence(fractions, min_fraction, n_ranked_colors, name, unit)

    if getattr(fractions, "name", None) != THRESHOLD_FRACTIONS:
        raise TypeError("color_presence reads a threshold_fractions(...) metric, got "
                        f"{getattr(fractions, 'name', type(fractions).__name__)!r}")

    colors = [spec["name"] for spec in fractions.parameters["thresholds"]]
    return derived(_color_presence, [fractions], from_run, name=name, unit=unit,
                   parameters={"colors": colors, "min_fraction": min_fraction,
                               "n_ranked_colors": n_ranked_colors})


def _color_presence(values, colors, min_fraction, n_ranked_colors):
    (stored,) = values.values()
    present = {color: stored[color] for color in colors
               if stored.get(color) is not None and stored[color] >= min_fraction}
    # sorted() is stable, so equal fractions keep the bins' order.
    ranked = sorted(present, key=lambda color: -present[color])
    result = {f"{color}_present": color in present for color in colors}
    result["n_colors_present"] = len(present)
    for rank in range(n_ranked_colors):
        result[f"ranked_color_{rank + 1}"] = ranked[rank] if rank < len(ranked) else None
    return result


class _UnboundColorPresence(DerivedMetric):
    """A `color_presence` that hasn't been given its `threshold_fractions`; `bind()` finds it in the run."""

    def __init__(self, fractions, min_fraction, n_ranked_colors, name, unit):
        # The named feature only marks this unbound; bind() does its own lookup.
        super().__init__(_color_presence, [fractions or THRESHOLD_FRACTIONS], name=name, unit=unit)
        self.fractions = fractions
        self.min_fraction = min_fraction
        self.n_ranked_colors = n_ranked_colors

    def bind(self, earlier):
        """
        The `color_presence` of the `threshold_fractions` listed before this one.

        - `earlier` -- the Metric operations listed before this one in the run.

        Returns the bound metric, the same one passing that operation builds.
        """
        candidates = [operation for operation in earlier
                      if getattr(operation, "name", None) == THRESHOLD_FRACTIONS]
        if self.fractions is None:
            if len(candidates) != 1:
                names = [operation.metric_name for operation in candidates]
                raise ValueError(
                    f"{self.metric_name} reads the threshold_fractions listed before it in "
                    f"metrics=, and found {names or 'none'} -- "
                    + ('say which with fractions="<its name>"' if names
                       else "list one first"))
            chosen = candidates[0]
        else:
            named = [operation for operation in earlier
                     if operation.metric_name == self.fractions]
            if not named:
                raise ValueError(
                    f"{self.metric_name} reads {self.fractions!r} from this run, but no "
                    "metric of that name is listed before it in metrics=")
            chosen = named[0]
        return color_presence(chosen, min_fraction=self.min_fraction,
                              n_ranked_colors=self.n_ranked_colors,
                              name=self.metric_name, unit=self.unit)


def threshold_masks(pixels, thresholds):
    """
    Which pixels match each colour threshold, converting to each colour space once.

    - `pixels` -- uint8 BGR, an `(N, 3)` pixel list or an `(H, W, 3)` image.
    - `thresholds` -- `ColorThreshold`s.

    Returns `{threshold name: boolean array}`, in the order given.
    """
    spaces = {space for threshold in thresholds for space in threshold.spaces}
    converted = {space: convert(pixels, space) for space in spaces}
    return {threshold.name: threshold.matches(converted) for threshold in thresholds}


def threshold_panel(segment, masks, fractions):
    """
    One picture per threshold, side by side: the pixels it matched in their own colour, the rest of the organism grey.

    - `segment` -- the segment measured.
    - `masks` -- `{name: boolean (N,)}` over the segment's masked pixels, as `threshold_masks` returns.
    - `fractions` -- `{name: fraction}`, annotated on each picture.

    Returns a uint8 BGR panel.
    """
    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    organism = segment.mask

    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    dimmed = (grey * _DIMMED).astype(np.uint8)

    pictures = []
    for name, matched in masks.items():
        picture = np.zeros_like(image)
        picture[organism] = dimmed[organism]
        selected = np.zeros(organism.shape, dtype=bool)
        selected[organism] = matched
        picture[selected] = image[selected]
        annotate(picture, f"{name} {fractions[name]:.1%}")
        pictures.append(picture)
    return side_by_side(*pictures)


def black_fraction(threshold=BLACK_THRESHOLD, name=None, unit="fraction"):
    """
    Metric: fraction of masked pixels darker than `threshold` lightness.

    Melanisation is the usual reason to want this, and it's a genuinely
    different question from mean lightness: a mostly-pale organism with heavy
    black markings and a uniformly mid-grey one can share a mean and differ
    completely here.

    - `threshold` -- lightness cutoff on a 0-1 scale.
    """
    # Rounded so a 0.2 cutoff is stored as 20.0 rather than 20.000000000000004.
    black = color_threshold("black", lab_l=(None, round(threshold * 100.0, 9)))
    return Metric("black_fraction", _single_fraction, {"threshold": black.spec()},
                  version="1", unit=unit, metric_name=name)


def hue_fraction(hue, min_saturation=MIN_SATURATION, min_value=MIN_VALUE,
                 name=None, unit="fraction"):
    """
    Metric: fraction of masked pixels falling in one named hue band.

    - `hue` -- one of `HUE_BANDS` (`"red"`, `"yellow"`, `"green"`, ...).
    - `min_saturation` -- pixels greyer than this are excluded rather than
      assigned a hue they don't really have.
    - `min_value` -- pixels darker than this are excluded for the same
      reason; hue is meaningless in shadow.
    """
    if hue not in HUE_BANDS:
        raise ValueError(f"unknown hue {hue!r} -- expected one of {sorted(HUE_BANDS)}")

    threshold = _hue_threshold(hue, min_saturation, min_value)
    return Metric(f"{hue}_fraction", _single_fraction, {"threshold": threshold.spec()},
                  version="1", unit=unit, metric_name=name)


def red_fraction(name=None, **kwargs):
    """Metric: fraction of masked pixels in the red hue band. See hue_fraction()."""
    return hue_fraction("red", name=name, **kwargs)


def yellow_fraction(name=None, **kwargs):
    """Metric: fraction of masked pixels in the yellow hue band. See hue_fraction()."""
    return hue_fraction("yellow", name=name, **kwargs)


def _hue_threshold(hue, min_saturation, min_value):
    return color_threshold(hue, hsv_h=HUE_BANDS[hue], hsv_s=(min_saturation, None), hsv_v=(min_value, None))


def _checked_condition(name, space, channel, low=None, high=None, *extra):
    """One `(space, channel, low, high)` row, validated against the colour space and made JSON-plain."""
    if extra:
        raise ValueError(f"colour threshold {name!r}: bounds for {space}.{channel} should be (low, high)")
    target = get_space(space)
    target.index(channel)
    low = None if low is None else float(low)
    high = None if high is None else float(high)

    where = f"colour threshold {name!r}, {space}.{channel}"
    if channel in target.circular:
        if low is None or high is None:
            raise ValueError(f"{where}: an arc on a circular channel needs both ends")
        if not (0.0 <= low <= 360.0 and 0.0 <= high <= 360.0):
            raise ValueError(f"{where}: arc ends must lie in 0-360 degrees, got ({low}, {high})")
        if low == high:
            raise ValueError(f"{where}: ({low}, {high}) is an empty arc; (0, 360) is the whole circle")
    else:
        if low is None and high is None:
            raise ValueError(f"{where}: needs at least one bound")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"{where}: low {low} is not below high {high}")
    return (target.name, channel, low, high)


def _measure(segment, thresholds, unmatched):
    pixels = masked_pixels(segment)
    masks = threshold_masks(pixels, thresholds)
    fractions = {name: float(matched.mean()) for name, matched in masks.items()}
    if unmatched:
        masks[UNMATCHED] = ~np.logical_or.reduce(list(masks.values()))
        fractions[UNMATCHED] = float(masks[UNMATCHED].mean())
    return masks, fractions


def _threshold_fractions(segment, thresholds, unmatched=False):
    thresholds = [ColorThreshold.from_spec(spec) for spec in thresholds]
    masks, fractions = _measure(segment, thresholds, unmatched)
    if segment.panel_sink is not None:
        segment.emit_panel(threshold_panel(segment, masks, fractions), "threshold_fractions")
    return fractions


def _single_fraction(segment, threshold):
    threshold = ColorThreshold.from_spec(threshold)
    masks, fractions = _measure(segment, [threshold], unmatched=False)
    if segment.panel_sink is not None:
        segment.emit_panel(threshold_panel(segment, masks, fractions), f"{threshold.name}_fraction")
    return fractions[threshold.name]
