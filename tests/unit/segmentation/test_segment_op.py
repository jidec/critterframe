"""
The segmenter contract, and what a run does with a model that breaks it.

`segment(model)` asks for `predict(image) -> (mask, score, info)` and nothing
else. That is what makes a hand-drawn mask, a threshold, and SAM2 alternative
segmentations rather than different systems -- and it is why every test in this
suite that needs a segmenter uses twenty lines of thresholding instead of a
GPU.

The guards matter more than the happy path. A mask of the wrong shape cannot be
aligned back to original coordinates, and an empty mask is a failed
segmentation rather than a zero-area organism; both would otherwise be persisted
as ordinary-looking masks.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Recipe, Segment
from critterframe.segmentation.run import _build_recipes
from helpers.models import (
    EmptyMaskModel,
    NoThresholdModel,
    ThresholdModel,
    UnidentifiedModel,
    WrongShapeModel,
)
from helpers.synthetic import draw_specimen


def a_segment(panel_sink=None):
    return Segment(draw_specimen(0), occurrence_id="test",
                   panel_sink=panel_sink)


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def test_a_model_s_mask_becomes_the_segment_s_mask():
    segmented, _info = cf.segment(ThresholdModel())(a_segment())
    assert segmented.mask is not None
    assert segmented.mask.shape == segmented.shape
    assert segmented.mask.any()


def test_the_run_records_what_the_model_reported():
    _segmented, info = cf.segment(ThresholdModel())(a_segment())
    assert info["score"] == 0.9
    assert info["area"] > 0
    assert 0 < info["area_fraction"] < 1
    assert info["cutoff"] == 100          # the model's own diagnostics survive


def test_a_model_without_a_score_reports_none_rather_than_zero():
    """
    Zero would be a confidence. "This segmenter doesn't produce one" is a
    different fact, and the column has to be able to say it.
    """
    _segmented, info = cf.segment(UnidentifiedModel())(a_segment())
    assert info["score"] is None


def test_a_model_that_takes_no_threshold_still_works():
    """
    Not every segmenter produces something thresholdable in the first place, so
    the call falls back rather than requiring every model to accept an argument
    it has no use for.
    """
    segmented, _info = cf.segment(NoThresholdModel())(a_segment())
    assert segmented.mask.any()


def test_a_mask_of_the_wrong_shape_is_refused():
    """
    It cannot be aligned back to original coordinates, so it must raise rather
    than be persisted somewhere plausible-looking.
    """
    with pytest.raises(ValueError, match="mask matching the image"):
        cf.segment(WrongShapeModel())(a_segment())


def test_an_empty_mask_is_refused():
    """A failed segmentation, not a zero-area organism."""
    with pytest.raises(ValueError, match="empty mask"):
        cf.segment(EmptyMaskModel())(a_segment())


def test_the_threshold_reaches_the_model_and_the_hash():
    """
    In whatever units that model thresholds in -- for SAM2 a logit, not a
    probability -- which is why it is passed through rather than normalized.
    """
    operation = cf.segment(ThresholdModel(), mask_threshold=0.5)
    assert operation.parameters == {"mask_threshold": 0.5}
    assert operation.spec() != cf.segment(ThresholdModel()).spec()


def test_the_model_reaches_the_hash_through_its_identity():
    assert (cf.segment(ThresholdModel(cutoff=100)).spec()
            != cf.segment(ThresholdModel(cutoff=120)).spec())


def test_a_segmentation_is_a_segment_kind_operation():
    """
    Which is what makes draw_mask() and segment(model) interchangeable in a
    recipe: same kind, same contract, different implementation.
    """
    assert cf.segment(ThresholdModel()).kind == "segment"
    assert cf.draw_mask().kind == "segment"


# ---------------------------------------------------------------------------
# Recipe construction
# ---------------------------------------------------------------------------


def test_one_part_gets_one_recipe():
    recipes = _build_recipes("segments", [cf.segment(ThresholdModel())], None,
                             None, "organism", None, False)
    assert list(recipes) == ["organism"]
    assert isinstance(recipes["organism"], Recipe)


def test_several_outputs_get_a_recipe_each():
    recipes = _build_recipes("parts", None,
                             {"head": [cf.segment(ThresholdModel())],
                              "wing": [cf.segment(ThresholdModel(cutoff=120))]},
                             None, "organism", None, False)
    assert sorted(recipes) == ["head", "wing"]
    assert recipes["head"].hash != recipes["wing"].hash


def test_shared_steps_go_in_front_of_every_output_s_own():
    """
    How a multi-output run forks one preprocessed segment into a branch per
    part -- and the shared work is part of each branch's identity, because it
    changed what that branch saw.
    """
    recipes = _build_recipes("parts", None,
                             {"head": [cf.segment(ThresholdModel())]},
                             [cf.remove_background()], "organism", None, False)
    assert [op.name for op in recipes["head"].operations] == [
        "remove_background", "segment"]


def test_the_mask_table_being_read_is_part_of_identity():
    """
    A recipe measuring reference masks is not the recipe measuring canonical
    ones, even with identical operations.
    """
    canonical = _build_recipes("segments", [cf.segment(ThresholdModel())], None,
                               None, "organism", None, False)["organism"]
    reference = _build_recipes("segments", [cf.segment(ThresholdModel())], None,
                               None, "organism", None, True)["organism"]
    assert canonical.hash != reference.hash


@pytest.mark.parametrize("kwargs, message", [
    ({"steps": [], "outputs": {}}, "either steps="),
    ({"steps": None, "outputs": None}, "needs steps="),
])
def test_steps_and_outputs_are_mutually_exclusive(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _build_recipes("segments", kwargs["steps"], kwargs["outputs"], None,
                       "organism", None, False)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.stages = []

    def collect(self, occurrence_id, stage, image):
        self.stages.append(stage)
        assert image.dtype == np.uint8


def test_the_operation_itself_draws_nothing():
    """
    The mask panel every segmentation run contributes is drawn by the RUN, not
    by the operation -- deliberately, because it is the one view that always
    exists, and a segmenter with no visualize() hook of its own would otherwise
    put nothing on the grid at exactly the moment you most want to look.
    (`tests/integration/test_visualization_outputs.py` checks the run side.)
    """
    sink = RecordingSink()
    cf.segment(ThresholdModel())(a_segment(panel_sink=sink))
    assert sink.stages == []


def test_a_model_may_draw_its_own_panel_instead():
    """
    `visualize(...)` on the model, for a segmenter whose decision is worth
    showing differently -- a detector's boxes, say. Only consulted when there
    is a sink to draw for.
    """
    class DrawsItsOwn(ThresholdModel):
        def visualize(self, segment, image, mask, score, info):
            segment.emit_panel(np.zeros_like(image), "custom")

    sink = RecordingSink()
    cf.segment(DrawsItsOwn())(a_segment(panel_sink=sink))
    assert sink.stages == ["custom"]


def test_no_panel_is_drawn_for_the_occurrences_nobody_is_looking_at():
    segmented, _info = cf.segment(ThresholdModel())(a_segment())
    assert segmented.mask.any()


# ---------------------------------------------------------------------------
# Manual segmentation shares the contract
# ---------------------------------------------------------------------------


def test_hand_drawn_and_automatic_masks_are_the_same_kind_of_thing():
    """
    What differs between draw_mask() and segment(model) is only which one a
    run's recipe names, and therefore which recipe hash the resulting mask
    carries.
    """
    drawn = Recipe("segment", "by_hand", [cf.draw_mask()], part="organism")
    automatic = Recipe("segment", "by_hand", [cf.segment(ThresholdModel())],
                       part="organism")
    assert drawn.hash != automatic.hash
    assert drawn.kind == automatic.kind == "segment"


def test_a_brush_size_is_part_of_the_recipe():
    assert cf.draw_mask(brush_radius=5).spec() != cf.draw_mask(brush_radius=9).spec()


def test_correcting_and_drawing_are_different_recipes():
    """
    One starts from the existing mask and one from nothing, so they are not
    interchangeable work even at the same brush size.
    """
    assert cf.correct_mask().spec() != cf.draw_mask().spec()
