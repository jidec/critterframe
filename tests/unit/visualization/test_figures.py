"""
Whole-population figures render headless and warning-free.

The suite turns every warning into an error, so a builder that trips a matplotlib layout warning
fails here rather than in the middle of someone's run.
"""

import numpy as np
import pytest

from critterframe.visualization import figures


BUILDERS = {
    "line": lambda: figures.line_chart({"train": [3, 2, 1], "val": ([0, 1, 2], [4, 3, 3])},
                                       title="loss", xlabel="epoch", marks={"best": 1}),
    "bars": lambda: figures.bar_chart({"a": 3, "b": 5}),
    "stacked": lambda: figures.bar_chart({"train": {"x": 3, "y": 1}, "val": {"x": 1}}),
    "side_by_side": lambda: figures.bar_chart({"train": {"x": 3, "y": 1}}, stacked=False),
    "histogram": lambda: figures.histogram({"clean": [0.1, 0.5, 0.9], "bad": [0.2, None]},
                                           marks={"cutoff": 0.4}),
    "empty_histogram": lambda: figures.histogram([]),
    "scatter": lambda: figures.scatter([1, 2, 3], [1.1, 2.2, 2.9], diagonal=True,
                                       groups=["a", "b", "a"]),
    "funnel": lambda: figures.funnel({"read": 100, "dropped": 90, "final": 70}),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_saves_a_non_empty_png(tmp_path, name):
    written = figures.save_figure(BUILDERS[name](), tmp_path / "sub" / f"{name}.png")
    assert written.exists()
    assert written.stat().st_size > 0


def test_a_figure_renders_as_a_display_ready_panel():
    image = figures.to_image(figures.funnel({"read": 3, "kept": 2}))
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
