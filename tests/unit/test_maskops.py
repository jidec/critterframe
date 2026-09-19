"""
The mask arithmetic every layer shares.

Pure functions over boolean arrays, so every case here is checked against a
number computed by hand. The empty cases are the ones worth pinning: two empty
masks AGREE (vacuously, 1.0) rather than dividing by zero, and something
against nothing agrees about nothing -- five copies of this arithmetic is five
chances for two of them to answer differently.
"""

import numpy as np
import pytest

from critterframe.maskops import (
    largest_component,
    mask_bounds,
    mask_coverage,
    mask_iou,
    pad_to_common_shape,
)


def a_mask(shape=(100, 100), box=(slice(20, 60), slice(20, 60))):
    mask = np.zeros(shape, bool)
    mask[box] = True
    return mask


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------


def test_identical_masks_agree_completely():
    score = mask_iou(a_mask(), a_mask())
    assert score == 1.0


def test_disjoint_masks_agree_about_nothing():
    other = a_mask(box=(slice(70, 90), slice(70, 90)))
    score = mask_iou(a_mask(), other)
    assert score == 0.0


def test_partial_overlap_is_intersection_over_union():
    """
    Two 40x40 boxes overlapping in a 20x40 strip: 800 shared out of 2400
    covered.
    """
    other = a_mask(box=(slice(20, 60), slice(40, 80)))
    score = mask_iou(a_mask(), other)
    assert score == pytest.approx(800 / 2400)


def test_two_empty_masks_agree():
    """
    Vacuously, and 1.0 is the right answer rather than a division by zero:
    neither found anything, and they do not disagree about where it is.
    """
    score = mask_iou(np.zeros((10, 10), bool),
                                        np.zeros((10, 10), bool))
    assert score == 1.0


def test_something_against_nothing_agrees_about_nothing():
    score = mask_iou(a_mask(), np.zeros((100, 100), bool))
    assert score == 0.0


def test_masks_of_different_sizes_are_padded_to_compare():
    """
    A shape mismatch means one was stored from a differently-sized frame. "They
    agree about this much" beats a crash.
    """
    assert mask_iou(a_mask(shape=(80, 80)), a_mask()) == 1.0


def test_padding_returns_both_masks_on_one_canvas():
    """What a caller drawing the comparison needs, so it draws what was scored."""
    padded, padded_reference = pad_to_common_shape(a_mask(shape=(80, 80)), a_mask())
    assert padded.shape == padded_reference.shape == (100, 100)


# ---------------------------------------------------------------------------
# mask_coverage
# ---------------------------------------------------------------------------


def test_identical_masks_have_full_coverage():
    assert mask_coverage(a_mask(), a_mask()) == 1.0


def test_disjoint_masks_have_no_coverage():
    other = a_mask(box=(slice(70, 90), slice(70, 90)))
    assert mask_coverage(a_mask(), other) == 0.0


def test_coverage_is_intersection_over_reference_area_only():
    """
    A 20x40 strip shared out of the reference's own 40x40 -- unlike IoU, the
    denominator is the reference alone, so area the mask covers outside the
    reference never appears in it.
    """
    reference = a_mask(box=(slice(20, 60), slice(40, 80)))
    assert mask_coverage(a_mask(), reference) == pytest.approx(800 / 1600)


def test_extra_predicted_area_outside_the_reference_costs_nothing():
    """
    The property `mask_iou` doesn't have: a mask several times larger than the
    reference, but that fully contains it, still scores 1.0 -- an organism
    mask that also picks up the wings shouldn't be penalized when all that
    matters downstream is that the body is inside it.
    """
    reference = a_mask(box=(slice(20, 60), slice(20, 60)))
    much_bigger = a_mask(box=(slice(0, 100), slice(0, 100)))
    assert mask_coverage(much_bigger, reference) == 1.0

    iou = mask_iou(much_bigger, reference)
    assert iou < 1.0


def test_empty_reference_has_vacuous_full_coverage():
    score = mask_coverage(a_mask(), np.zeros((100, 100), bool))
    assert score == 1.0


def test_coverage_pads_masks_of_different_sizes():
    small_reference = a_mask(shape=(80, 80))
    score = mask_coverage(a_mask(), small_reference)
    assert score == 1.0


# ---------------------------------------------------------------------------
# mask_bounds
# ---------------------------------------------------------------------------


def test_bounds_are_inclusive_of_the_last_row_and_column():
    """So width is what a crop of that box would actually be."""
    assert mask_bounds(a_mask()) == {"x": 20, "y": 20, "width": 40, "height": 40}


def test_a_single_pixel_is_one_by_one():
    assert mask_bounds(a_mask(box=(slice(5, 6), slice(7, 8)))) == {
        "x": 7, "y": 5, "width": 1, "height": 1}


def test_an_empty_mask_has_no_box_to_report():
    with pytest.raises(ValueError, match="empty mask"):
        mask_bounds(np.zeros((10, 10), bool))


# ---------------------------------------------------------------------------
# largest_component
# ---------------------------------------------------------------------------


def test_the_biggest_blob_survives_and_the_rest_do_not():
    mask = a_mask()
    mask[80:83, 80:83] = True          # a speck, far from the body

    kept = largest_component(mask)
    assert kept[20:60, 20:60].all()
    assert not kept[80:83, 80:83].any()


def test_one_blob_comes_back_unchanged():
    assert (largest_component(a_mask()) == a_mask()).all()
