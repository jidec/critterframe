"""
One picture of one decision, and the colour conventions they all share.

Learning to read one panel should be learning to read all of them, which is why
the drawing helpers live in one module rather than being reimplemented per
operation: red always means removed, yellow and magenta always mean "only this
one has it". A helper that drew disagreement in a different colour per operation
would make every panel need its own legend.

The other thing under test here is `PanelFiles` -- the sink that writes
full-resolution files per panel. It is deliberately NOT reachable through a
run's `visualize=`, because a 10,000-occurrence run cannot be inspected as
10,000 files; you reach for it when you have built one segment by hand.
"""

import numpy as np
import pytest

from critterframe.project import paths
from critterframe.visualization.panels import (
    PanelFiles,
    annotate,
    diff_panel,
    mask_to_bgr,
    overlay_mask,
    save_panel,
    side_by_side,
)


def a_mask(shape=(40, 60)):
    mask = np.zeros(shape, bool)
    mask[10:30, 20:40] = True
    return mask


def an_image(shape=(40, 60), value=120):
    return np.full((*shape, 3), value, np.uint8)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def test_a_mask_becomes_a_viewable_image():
    panel = mask_to_bgr(a_mask())
    assert panel.shape == (40, 60, 3)
    assert panel.dtype == np.uint8
    assert set(np.unique(panel)) <= {0, 255}


def test_an_overlay_keeps_the_image_visible_underneath():
    """
    Half-transparent on purpose: a solid fill would hide exactly the boundary
    you are trying to judge.
    """
    overlaid = overlay_mask(an_image(), a_mask())
    inside = overlaid[20, 30]
    outside = overlaid[0, 0]

    assert not np.array_equal(inside, outside)
    assert outside.tolist() == [120, 120, 120]
    assert overlaid.dtype == np.uint8


def test_an_overlay_does_not_modify_the_image_it_was_given():
    """
    Operations pass their working image straight in; a helper that drew on it
    would corrupt the segment for every step afterwards.
    """
    image = an_image()
    overlay_mask(image, a_mask())
    assert (image == 120).all()


def test_a_diff_panel_separates_agreement_from_each_side_s_own():
    """
    Three colours, three meanings: both, only the first, only the second. It is
    what makes an IoU number checkable by eye.
    """
    first, second = a_mask(), a_mask()
    second[10:30, 30:50] = True             # overlaps and extends right

    panel = diff_panel(first, second)
    colours = {tuple(colour) for colour in panel.reshape(-1, 3)}
    assert len(colours) >= 3


def test_annotating_writes_on_the_panel_in_place():
    panel = an_image(value=0)
    annotate(panel, "body_length 120px")
    assert panel.any()


def test_two_lines_of_annotation_do_not_overwrite_each_other():
    one_line = an_image(value=0)
    annotate(one_line, "first")
    two_lines = an_image(value=0)
    annotate(two_lines, "first")
    annotate(two_lines, "second", line=1)

    assert two_lines.sum() > one_line.sum()


def test_side_by_side_joins_panels_of_different_heights():
    """
    Before-and-after panels are rarely the same size once something has been
    cropped, and the comparison is the point.
    """
    joined = side_by_side(an_image((40, 60)), an_image((80, 30)))
    assert joined.shape[1] >= 90
    assert joined.dtype == np.uint8


def test_side_by_side_accepts_masks_and_images_together():
    joined = side_by_side(a_mask(), an_image())
    assert joined.ndim == 3


# ---------------------------------------------------------------------------
# save_panel / PanelFiles
# ---------------------------------------------------------------------------


def test_a_panel_is_written_where_it_was_asked_for(tmp_path):
    written = save_panel(tmp_path, an_image(), "specimen0", subdir="scale")
    assert written.exists()
    assert written.parent == paths.visualizations_dir(tmp_path, "scale")


def test_saving_creates_the_directory_it_needs(tmp_path):
    """Lazily, like every other writer -- nothing exists until something writes."""
    assert not paths.visualizations_dir(tmp_path).exists()
    save_panel(tmp_path, an_image(), "specimen0")
    assert paths.visualizations_dir(tmp_path).exists()


def test_panel_files_writes_one_file_per_stage(tmp_path):
    sink = PanelFiles(tmp_path)
    sink.collect("specimen0", "orientation", an_image())
    sink.collect("specimen0", "crop_to_mask", an_image())

    written = list(paths.visualizations_dir(tmp_path).rglob("*.png"))
    assert len(written) == 2


def test_panel_files_wants_every_occurrence(tmp_path):
    """
    Unlike a run's report, which wants only its sample: this sink exists for
    the case where you built the segment yourself and every panel is wanted.
    """
    sink = PanelFiles(tmp_path)
    assert sink.wants("anything") is True


def test_full_resolution_is_the_point(tmp_path):
    """
    A grid cell is a thumbnail. When you need to see whether a boundary is one
    pixel out, you need the pixels.
    """
    import cv2

    sink = PanelFiles(tmp_path)
    sink.collect("specimen0", "mask", an_image((400, 600)))
    written = next(iter(paths.visualizations_dir(tmp_path).rglob("*.png")))
    assert cv2.imread(str(written)).shape[:2] == (400, 600)


@pytest.mark.parametrize("panel", [np.zeros((4, 4), bool),
                                   np.zeros((4, 4), np.uint8),
                                   np.zeros((4, 4, 3), np.uint8)])
def test_any_display_ready_panel_can_be_saved(tmp_path, panel):
    assert save_panel(tmp_path, panel, "specimen0").exists()
