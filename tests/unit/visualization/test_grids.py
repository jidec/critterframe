"""
Laying many panels out as one image. Pure geometry, no project, no I/O.

The rule that shapes the module is at the top of `as_bgr`: panels must ARRIVE
display-ready. A grid will not rescale a float array, because two probability
maps with different ranges would stretch to look identical and the grid would
have quietly invented a comparison. Only the operation that computed the numbers
knows what they mean, so rendering happens there.

The second rule is in `fit_cell`: images scale by one factor in both axes and
letterbox. A cell that stretched to fill would make a long specimen and a round
one look alike, which is precisely the judgement a QC grid exists to support.
"""

import numpy as np
import pytest

from critterframe.visualization.grids import (
    as_bgr,
    comparison_grid,
    fit_cell,
    image_grid,
)


def panel(height=40, width=60, value=200):
    return np.full((height, width, 3), value, np.uint8)


# ---------------------------------------------------------------------------
# as_bgr
# ---------------------------------------------------------------------------


def test_a_boolean_mask_becomes_black_and_white():
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    converted = as_bgr(mask)

    assert converted.shape == (4, 4, 3)
    assert converted[0, 0].tolist() == [255, 255, 255]
    assert converted[1, 1].tolist() == [0, 0, 0]


def test_a_grayscale_panel_becomes_three_channels():
    assert as_bgr(np.zeros((4, 4), np.uint8)).shape == (4, 4, 3)


def test_alpha_is_dropped():
    assert as_bgr(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)


def test_a_bgr_panel_passes_through():
    original = panel()
    assert as_bgr(original) is original


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.int32])
def test_anything_that_is_not_display_ready_raises(dtype):
    """
    And the message says where to fix it: render it where it's made. A grid
    that guessed would silently make two incomparable probability maps look
    like a comparison.
    """
    with pytest.raises(TypeError, match="not uint8"):
        as_bgr(np.zeros((4, 4), dtype))


# ---------------------------------------------------------------------------
# fit_cell
# ---------------------------------------------------------------------------


def test_a_cell_is_exactly_the_size_asked_for():
    assert fit_cell(panel(400, 800), cell=(100, 100)).shape == (100, 100, 3)
    assert fit_cell(panel(10, 10), cell=(100, 100)).shape == (100, 100, 3)


def test_aspect_ratio_survives_the_fit():
    """
    A wide panel letterboxes with background above and below rather than being
    squashed -- so a long specimen still looks long in the grid.
    """
    fitted = fit_cell(panel(20, 100, value=255), cell=(100, 100), background=0)
    lit_rows = np.unique(np.nonzero(fitted.any(axis=(1, 2)))[0])

    assert len(lit_rows) < 100
    assert fitted[0].sum() == 0 and fitted[-1].sum() == 0


def test_an_empty_panel_becomes_a_blank_cell():
    """
    Rather than raising: one missing panel in a grid of twenty-five is a gap to
    look at, not a reason to lose the other twenty-four.
    """
    blank = fit_cell(np.zeros((0, 0, 3), np.uint8), cell=(50, 50), background=17)
    assert blank.shape == (50, 50, 3)
    assert (blank == 17).all()


# ---------------------------------------------------------------------------
# image_grid
# ---------------------------------------------------------------------------


def test_a_grid_is_one_image_holding_them_all():
    grid = image_grid([panel(), panel(), panel()], columns=2, cell=(50, 50))
    assert grid.dtype == np.uint8
    assert grid.shape[2] == 3
    assert grid.shape[0] >= 100 and grid.shape[1] >= 100


def test_ragged_input_sizes_are_fine():
    """
    Which they always are in practice: every specimen's crop is a different
    shape, and a grid that required uniform input would be unusable.
    """
    grid = image_grid([panel(20, 300), panel(300, 20), panel(64, 64)],
                      columns=3, cell=(80, 80))
    assert grid.shape[0] >= 80


def test_labels_and_a_title_add_space_rather_than_covering_the_image():
    plain = image_grid([panel()], columns=1, cell=(50, 50))
    labelled = image_grid([panel()], labels=["specimen0"], columns=1,
                          cell=(50, 50))
    titled = image_grid([panel()], title="a run", columns=1, cell=(50, 50))

    assert labelled.shape[0] > plain.shape[0]
    assert titled.shape[0] > plain.shape[0]


def test_a_grid_needs_something_to_show():
    with pytest.raises(ValueError, match="at least one image"):
        image_grid([])


def test_labels_must_line_up_with_images():
    """
    A mislabelled QC grid is worse than an unlabelled one -- you would act on
    the wrong specimen.
    """
    with pytest.raises(ValueError):
        image_grid([panel(), panel()], labels=["only one"])


def test_a_boolean_mask_can_be_laid_out_directly():
    """Masks are the most common thing a segmentation grid shows."""
    grid = image_grid([np.ones((30, 30), bool)], columns=1, cell=(40, 40))
    assert grid.shape[1] == 40
    assert grid.dtype == np.uint8
    assert grid.max() == 255           # the mask is there, in white


# ---------------------------------------------------------------------------
# comparison_grid
# ---------------------------------------------------------------------------


def test_a_comparison_is_a_row_per_specimen_and_a_column_per_stage():
    """
    The layout that shows WHERE in a recipe something went wrong, as opposed to
    which specimen it went wrong on.
    """
    rows = [[panel(), panel(), panel()], [panel(), panel(), panel()]]
    grid = comparison_grid(rows, cell=(50, 50))

    assert grid.shape[0] >= 100        # two rows
    assert grid.shape[1] >= 150        # three columns


def test_column_titles_and_row_labels_fit_around_it():
    rows = [[panel(), panel()]]
    plain = comparison_grid(rows, cell=(50, 50))
    titled = comparison_grid(rows, column_titles=["before", "after"],
                             row_labels=["specimen0"], cell=(50, 50))
    assert titled.shape[0] > plain.shape[0]


def test_a_missing_panel_in_a_row_leaves_a_gap():
    """
    An operation that didn't run for one specimen is a hole in that row, and
    seeing the hole is the point -- the alternative is a shifted row where
    every column afterwards is mislabelled.
    """
    grid = comparison_grid([[panel(), None, panel()]], cell=(50, 50))
    assert grid.shape[1] >= 150


def test_a_comparison_needs_at_least_one_row():
    with pytest.raises(ValueError, match="at least one row"):
        comparison_grid([])


def test_rows_of_different_lengths_are_padded_not_misaligned():
    """
    Otherwise a run where one specimen skipped a stage would put its last panel
    under the wrong column heading.
    """
    grid = comparison_grid([[panel(), panel()], [panel()]],
                           column_titles=["a", "b"], cell=(50, 50))
    assert grid.shape[0] >= 100
