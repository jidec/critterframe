"""
Whole-population figures: line_chart, bar_chart, histogram, scatter, funnel.

Each builder returns a matplotlib `Figure` drawn on its own Agg canvas. pyplot is never imported,
so building one needs no display and changes no global backend. matplotlib is imported on first
use, so importing the package doesn't pay for it.
"""

from pathlib import Path

import numpy as np

DEFAULT_SIZE = (6.4, 4.0)
DPI = 100


def _new_figure(title=None, xlabel=None, ylabel=None, size=DEFAULT_SIZE):
    """A blank figure with one axes, on an Agg canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=size, dpi=DPI, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    if title:
        axes.set_title(title)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    return figure, axes


def _mark(axes, marks, vertical=True):
    """Dashed reference lines at named positions, e.g. a suggested threshold."""
    for label, position in (marks or {}).items():
        if position is None:
            continue
        line = axes.axvline if vertical else axes.axhline
        line(position, linestyle="--", color="gray", linewidth=1)
        axes.annotate(str(label), xy=(position, 1) if vertical else (1, position),
                      xycoords=("data", "axes fraction") if vertical
                      else ("axes fraction", "data"),
                      xytext=(3, -12) if vertical else (-3, 3),
                      textcoords="offset points", fontsize=8, color="gray",
                      ha="left" if vertical else "right")


def line_chart(series, title=None, xlabel=None, ylabel=None, marks=None):
    """
    One or more lines on shared axes, e.g. loss per epoch.

    - `series` -- `{label: ys}` or `{label: (xs, ys)}`. Bare ys are plotted
      against 0, 1, 2, ...
    - `title`, `xlabel`, `ylabel` -- text for the axes.
    - `marks` -- `{label: x}` vertical reference lines, e.g. the best epoch.

    Returns a Figure.
    """
    figure, axes = _new_figure(title, xlabel, ylabel)
    for label, values in series.items():
        if isinstance(values, tuple) and len(values) == 2:
            xs, ys = values
        else:
            ys = list(values)
            xs = range(len(ys))
        axes.plot(list(xs), list(ys), marker="o" if len(list(ys)) < 30 else None,
                  markersize=3, label=str(label))
    if len(series) > 1:
        axes.legend(fontsize=8)
    _mark(axes, marks)
    return figure


def bar_chart(counts, title=None, xlabel=None, ylabel=None, stacked=True):
    """
    Bars per category, or per group split by category.

    - `counts` -- `{category: value}` for one set of bars, or
      `{group: {category: value}}` for one bar per group, e.g. class counts
      per split.
    - `title`, `xlabel`, `ylabel` -- text for the axes.
    - `stacked` -- stack categories within a group rather than placing them
      side by side. Ignored for a flat `counts`.

    Returns a Figure.
    """
    figure, axes = _new_figure(title, xlabel, ylabel)
    groups = list(counts)
    nested = bool(groups) and isinstance(counts[groups[0]], dict)

    if not nested:
        positions = np.arange(len(groups))
        axes.bar(positions, [counts[group] for group in groups])
        axes.set_xticks(positions)
        axes.set_xticklabels([str(group) for group in groups], rotation=30, ha="right")
        return figure

    categories = sorted({category for group in groups for category in counts[group]},
                        key=str)
    positions = np.arange(len(groups))
    width = 0.8 if stacked else 0.8 / max(1, len(categories))
    bottom = np.zeros(len(groups))
    for index, category in enumerate(categories):
        values = np.array([counts[group].get(category, 0) for group in groups], float)
        if stacked:
            axes.bar(positions, values, width, bottom=bottom, label=str(category))
            bottom += values
        else:
            axes.bar(positions - 0.4 + width * (index + 0.5), values, width,
                     label=str(category))
    axes.set_xticks(positions)
    axes.set_xticklabels([str(group) for group in groups], rotation=30, ha="right")
    if len(categories) <= 20:
        axes.legend(fontsize=7, ncol=2)
    return figure


def histogram(values, bins=30, title=None, xlabel=None, ylabel="count", marks=None):
    """
    The distribution of one or more sets of values, overlaid.

    - `values` -- a sequence, or `{label: sequence}` to overlay several, e.g.
      clean vs flagged specimens.
    - `bins` -- number of bins, shared across every set.
    - `title`, `xlabel`, `ylabel` -- text for the axes.
    - `marks` -- `{label: x}` vertical reference lines, e.g. a threshold.

    Returns a Figure.
    """
    figure, axes = _new_figure(title, xlabel, ylabel)
    sets = values if isinstance(values, dict) else {None: values}
    finite = {label: np.asarray([v for v in data if v is not None], float)
              for label, data in sets.items()}
    finite = {label: data[np.isfinite(data)] for label, data in finite.items()}
    pooled = np.concatenate([data for data in finite.values()]) if finite else np.array([])

    if pooled.size:
        edges = np.histogram_bin_edges(pooled, bins=bins)
        for label, data in finite.items():
            axes.hist(data, bins=edges, alpha=0.6 if len(finite) > 1 else 1.0,
                      label=None if label is None else str(label))
        if len(finite) > 1:
            axes.legend(fontsize=8)
    _mark(axes, marks)
    return figure


def scatter(xs, ys, title=None, xlabel=None, ylabel=None, diagonal=False, groups=None):
    """
    One point per pair, e.g. predicted against reference values.

    - `xs`, `ys` -- equal-length sequences.
    - `title`, `xlabel`, `ylabel` -- text for the axes.
    - `diagonal` -- draw y = x, for comparing two measurements of one thing.
    - `groups` -- optional label per point, coloured by group.

    Returns a Figure.
    """
    figure, axes = _new_figure(title, xlabel, ylabel)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    if groups is None:
        axes.scatter(xs, ys, s=10)
    else:
        groups = np.asarray([str(group) for group in groups])
        for group in sorted(set(groups)):
            chosen = groups == group
            axes.scatter(xs[chosen], ys[chosen], s=10, label=group)
        if len(set(groups)) <= 20:
            axes.legend(fontsize=7)

    finite = np.isfinite(xs) & np.isfinite(ys)
    if diagonal and finite.any():
        low = float(min(xs[finite].min(), ys[finite].min()))
        high = float(max(xs[finite].max(), ys[finite].max()))
        axes.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
    return figure


def funnel(stages, title=None, xlabel="rows"):
    """
    Counts through an ordered sequence of stages, e.g. rows read, dropped, kept.

    - `stages` -- `{stage: count}` in order, first stage at the top.
    - `title`, `xlabel` -- text for the axes.

    Returns a Figure.
    """
    figure, axes = _new_figure(title, xlabel)
    names = [str(stage) for stage in stages]
    counts = [stages[stage] for stage in stages]
    positions = np.arange(len(names))[::-1]
    axes.barh(positions, counts)
    axes.set_yticks(positions)
    axes.set_yticklabels(names)
    for position, count in zip(positions, counts):
        axes.annotate(f"{count:,}", xy=(count, position), xytext=(3, 0),
                      textcoords="offset points", va="center", fontsize=8)
    axes.margins(x=0.15)
    return figure


def to_image(figure):
    """
    A figure rendered as a display-ready uint8 BGR array, so it can sit in a grid like a panel.

    - `figure` -- a Figure from one of the builders above.

    Returns an (h, w, 3) uint8 array.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = figure.canvas if isinstance(figure.canvas, FigureCanvasAgg) \
        else FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, [2, 1, 0]])


def save_figure(figure, path):
    """
    Write a figure as a PNG and return the path.

    - `figure` -- a Figure from one of the builders above.
    - `path` -- destination; its parent directory is created if missing.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(figure.canvas, FigureCanvasAgg):
        FigureCanvasAgg(figure)
    figure.savefig(str(path), dpi=DPI)
    return path
