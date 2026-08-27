"""
The bundled segmenter's identity -- tested without torch, which is the point.

`import critterframe` must work without the `[segmentation]` extra, so every
torch and transformers import in this module is deferred inside a method.
Constructing a `GroundedSAM2`, reading its `identity()`, and hashing a recipe
that names it therefore all work on a machine that has never seen a GPU -- and
those are exactly the parts worth testing here, because `identity()` reaches the
recipe hash of every mask this model will ever produce.

What it MEANS is the reason for the branch: a centre-point prompt and a detected
box are different segmentations of the same image, so the two configurations
must not hash alike. Actually running the weights is two tests, marked `gpu`,
deselected by default -- past that point you are testing SAM2 rather than
CritterFrame.
"""

import sys

import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Recipe
from critterframe.segmentation.groundedsam import GroundedSAM2


# ---------------------------------------------------------------------------
# Construction without torch
# ---------------------------------------------------------------------------


def test_constructing_the_model_loads_no_weights():
    """
    __init__ only stores configuration. A recipe naming SAM2 can be built,
    hashed, and inspected on a laptop with no GPU and no checkpoint downloaded.
    """
    model = GroundedSAM2()
    assert model.model is None
    assert model.detector is None


def test_constructing_the_model_imports_no_torch():
    """
    The standing rule, at the one place most likely to break it. If this fails,
    an import moved from inside a method to the top of the module.
    """
    for name in ("torch", "transformers"):
        sys.modules.pop(name, None)

    cf.groundedsam2()
    cf.sam2()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_a_recipe_naming_it_can_be_hashed_without_torch():
    recipe = Recipe("segment", "organisms", [cf.segment(cf.groundedsam2())],
                    part="organism")
    assert len(recipe.hash) == 16


# ---------------------------------------------------------------------------
# identity()
# ---------------------------------------------------------------------------


def test_the_checkpoint_is_the_important_part():
    """
    Two runs of "sam2" against different weights are not equivalent work and
    must not be mistaken for it.
    """
    assert (cf.groundedsam2(model_name="a").identity()
            != cf.groundedsam2(model_name="b").identity())


def test_the_prompting_strategy_is_in_the_identity_too():
    """
    A detected box and a centre point are different segmentations of the same
    image, so the two configurations describe different work.
    """
    detected = cf.groundedsam2(detect_bounds=True).identity()
    prompted = cf.groundedsam2(detect_bounds=False).identity()

    assert detected != prompted
    assert "text_prompt" in detected
    assert "use_center_point" in prompted


def test_the_detector_settings_only_appear_when_a_detector_runs():
    """
    Otherwise changing a text prompt on a model that never detects anything
    would invalidate every mask it produced.
    """
    prompted = cf.sam2().identity()
    assert "detector" not in prompted
    assert "text_threshold" not in prompted


@pytest.mark.parametrize("kwargs", [
    {"text_prompt": "moth."},
    {"detector_name": "another-detector"},
    {"box_threshold": 0.5},
    {"text_threshold": 0.5},
    {"size": 512},
])
def test_every_setting_that_changes_the_mask_changes_the_identity(kwargs):
    assert cf.groundedsam2(**kwargs).identity() != cf.groundedsam2().identity()


@pytest.mark.parametrize("kwargs", [
    {"use_center_point": True},
    {"use_corner_points": True},
    {"retry_without_center": False},
    {"min_area_frac": 0.5},
])
def test_the_point_prompt_settings_change_it_on_the_other_branch(kwargs):
    assert cf.sam2(**kwargs).identity() != cf.sam2().identity()


def test_the_device_is_not_part_of_the_identity():
    """
    Running the same weights on CPU and on GPU is the same work, and a project
    processed on both must not recompute half of itself.
    """
    assert cf.groundedsam2(device="cpu").identity() == \
        cf.groundedsam2(device="cuda").identity()


def test_the_implementation_version_is_carried_explicitly():
    """
    Bumped by hand when this class's output changes for unchanged settings --
    which is the only way masks derived the old way can stop counting as work
    already done.
    """
    assert cf.groundedsam2().identity()["version"] == "2"


def test_sam2_is_groundedsam2_without_the_detector():
    """Shorthand, not a different model -- so the two hash alike."""
    assert cf.sam2().identity() == cf.groundedsam2(detect_bounds=False).identity()


def test_the_shorthand_can_still_be_told_to_detect():
    """`setdefault`, not an override: the caller has the last word."""
    assert cf.sam2(detect_bounds=True).detect_bounds is True


# ---------------------------------------------------------------------------
# The point-prompt geometry, which needs no model either
# ---------------------------------------------------------------------------


def test_a_centre_point_is_positive_and_the_corners_are_negative():
    """
    The geometric prompt used on a pre-cropped image: the organism is in the
    middle and the corners are background, which is what rejects shadows and
    clutter.
    """
    model = cf.sam2(use_center_point=True, use_corner_points=True)
    points, labels = model._prompt_points(200, 100, True, True)

    assert list(labels).count(1) == 1
    assert list(labels).count(0) == 4
    assert [100, 50] in [list(point) for point in points]


def test_no_prompt_at_all_is_allowed():
    """
    SAM2 segments the whole frame's dominant object with no prompt, which is
    the fallback when a centre point found almost nothing.
    """
    points, labels = cf.sam2()._prompt_points(200, 100, False, False)
    assert list(points) == [] and list(labels) == []


# ---------------------------------------------------------------------------
# Actually running the weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_model_segments_a_real_image(draw_specimen):
    """
    Deselected by default. The realistic assertion is that a mask of the right
    shape comes back -- anything more specific is testing SAM2, not this
    package.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    image = draw_specimen(0)
    mask, score, info = cf.sam2(use_center_point=True).predict(
        image[..., ::-1])         # RGB, as segment() passes it

    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool or set(np.unique(mask)) <= {0, 1}
    assert score is None or 0 <= float(score) <= 1
    assert isinstance(info, dict)


@pytest.mark.gpu
def test_the_model_runs_through_a_real_run(image_project):
    pytest.importorskip("torch")

    result = cf.run_segments(image_project, steps=[cf.segment(cf.sam2())],
                             limit=2, visualize=False)["organism"]
    assert result["processed"] + result["failed"] == 2
