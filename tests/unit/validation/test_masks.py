"""
Comparing masks against a reference. Nothing here persists anything.

The word "reference" is deliberate and is the thing to keep straight: a
reference is whatever you chose to compare against, and calling it ground truth
would assert the answer that validation exists to measure. A hand-corrected mask
is a reference; so is a second model's output; so is a mask from a slower,
better pipeline.

`mask_iou` pads to a common shape rather than refusing mismatched inputs,
because the comparison is between masks of the same occurrence and a shape
difference means one of them was stored from a differently-sized frame -- an
answer of "these agree about this much" is more useful there than a crash.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from critterframe.validation.masks import mask_iou
from helpers.models import ThresholdModel


def a_mask(shape=(100, 100), box=(slice(20, 60), slice(20, 60))):
    mask = np.zeros(shape, bool)
    mask[box] = True
    return mask


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------


def test_identical_masks_agree_completely():
    score, _mask, _reference = mask_iou(a_mask(), a_mask())
    assert score == 1.0


def test_disjoint_masks_agree_about_nothing():
    other = a_mask(box=(slice(70, 90), slice(70, 90)))
    score, _mask, _reference = mask_iou(a_mask(), other)
    assert score == 0.0


def test_partial_overlap_is_intersection_over_union():
    """
    Two 40x40 boxes overlapping in a 20x40 strip: 800 shared out of 2400
    covered.
    """
    other = a_mask(box=(slice(20, 60), slice(40, 80)))
    score, _mask, _reference = mask_iou(a_mask(), other)
    assert score == pytest.approx(800 / 2400)


def test_two_empty_masks_agree():
    """
    Vacuously, and 1.0 is the right answer rather than a division by zero:
    neither found anything, and they do not disagree about where it is.
    """
    score, _mask, _reference = mask_iou(np.zeros((10, 10), bool),
                                        np.zeros((10, 10), bool))
    assert score == 1.0


def test_something_against_nothing_agrees_about_nothing():
    score, _mask, _reference = mask_iou(a_mask(), np.zeros((100, 100), bool))
    assert score == 0.0


def test_masks_of_different_sizes_are_padded_to_compare():
    """
    A shape mismatch means one was stored from a differently-sized frame. "They
    agree about this much" beats a crash, and the padded pair comes back so a
    caller can see what was compared.
    """
    small = a_mask(shape=(80, 80))
    score, padded, padded_reference = mask_iou(small, a_mask())

    assert padded.shape == padded_reference.shape == (100, 100)
    assert score == 1.0


# ---------------------------------------------------------------------------
# validate_masks
# ---------------------------------------------------------------------------


@pytest.fixture
def with_reference(segmented_project):
    """
    A project whose canonical masks came from one segmenter and whose reference
    masks came from a stricter one -- the ordinary validation setup, where the
    two genuinely disagree.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, visualize=False)
    return segmented_project


@pytest.mark.slow
def test_validation_compares_the_two_tables(with_reference):
    scores = cf.validate_masks(with_reference, visualize=False)

    assert len(scores) == 8
    # Indexed by occurrence so a caller can look one up, sort by agreement, or
    # join it onto an export without renaming anything.
    assert scores.index.name == "occurrence_id"
    assert scores.columns.tolist() == ["iou"]
    assert (scores["iou"] > 0).all()
    assert (scores["iou"] < 1).all()       # a stricter segmenter really differs


@pytest.mark.slow
def test_an_occurrence_with_no_reference_is_left_out(segmented_project):
    """
    Reference masks are expensive -- somebody drew them -- so a project has
    them for a handful of occurrences, and validation reports on that handful
    rather than on the whole project.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, limit=3, visualize=False)

    assert len(cf.validate_masks(segmented_project, visualize=False)) == 3


@pytest.mark.slow
def test_validation_persists_nothing(with_reference):
    """
    All comparison, nothing stored. A validation score is a judgement about a
    method rather than a property of an organism, and storing it as a metric
    would put it in the trait table.
    """
    from critterframe.records.runs import load_runs

    before_runs = len(load_runs(with_reference))
    before_masks = len(mask_records.load_masks(with_reference))

    cf.validate_masks(with_reference, visualize=False)

    assert len(load_runs(with_reference)) == before_runs
    assert len(mask_records.load_masks(with_reference)) == before_masks
    assert len(cf.export_metrics(with_reference)) == 0


@pytest.mark.slow
def test_transforms_are_applied_to_both_sides(with_reference):
    """
    Otherwise the comparison would measure the transform rather than the
    disagreement -- which is why the chain is applied to the reference too.
    """
    plain = cf.validate_masks(with_reference, visualize=False)
    cleaned = cf.validate_masks(with_reference,
                                transforms=[cf.remove_appendages()],
                                visualize=False)
    assert len(plain) == len(cleaned)


@pytest.mark.slow
def test_a_project_with_no_reference_masks_has_nothing_to_validate(
        segmented_project, caplog):
    with caplog.at_level("WARNING"):
        scores = cf.validate_masks(segmented_project, visualize=False)
    assert len(scores) == 0
