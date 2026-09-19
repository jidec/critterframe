"""
The bundled segmenter's identity -- tested without torch, which is the point.

`import critterframe` must work without the `[torch]` extra, so every
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
import types

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
# _load(), with a faked transformers so no real weights are needed
# ---------------------------------------------------------------------------


class _FakeWeights:
    def to(self, device):
        return self


def _fake_transformers(detector_from_pretrained):
    """A minimal fake `transformers` module: SAM2 always loads; the detector's
    from_pretrained is whatever the caller supplies, so a test can make it
    fail once and succeed the next time."""
    class FakeSam2Processor:
        @staticmethod
        def from_pretrained(name):
            return types.SimpleNamespace(image_processor=types.SimpleNamespace(size=None))

    class FakeSam2Model:
        @staticmethod
        def from_pretrained(name):
            return _FakeWeights()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(name):
            return object()

    class FakeDetector:
        from_pretrained = staticmethod(detector_from_pretrained)

    return types.SimpleNamespace(
        Sam2Model=FakeSam2Model, Sam2Processor=FakeSam2Processor,
        AutoModelForZeroShotObjectDetection=FakeDetector,
        AutoProcessor=FakeAutoProcessor,
    )


def test_a_failed_detector_load_is_retried_not_skipped_forever(monkeypatch):
    """
    _load() used to gate everything on `self.model is not None` -- if SAM2
    itself loaded fine but the detector's load failed, self.model would stay
    set and the old guard would skip retrying the detector forever. Each
    piece must retry independently until it actually loads.
    """
    attempts = {"n": 0}

    def detector_from_pretrained(name):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("the paging file is too small")
        return _FakeWeights()

    monkeypatch.setitem(sys.modules, "transformers",
                        _fake_transformers(detector_from_pretrained))

    model = GroundedSAM2(detect_bounds=True, device="cpu")

    with pytest.raises(RuntimeError, match="paging file"):
        model._load()
    assert model.model is not None     # SAM2 itself loaded fine
    assert model.detector is None      # the failed load left nothing behind

    model._load()                       # retries only the missing piece
    assert model.detector is not None
    assert attempts["n"] == 2


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


# ---------------------------------------------------------------------------
# The detection prompt's form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", ["Dragonfly.", "dragonfly"])
def test_a_prompt_grounding_dino_reads_badly_is_warned_about(prompt, caplog):
    with caplog.at_level("WARNING"):
        model = cf.groundedsam2(text_prompt=prompt)
    assert "lowercase phrase ending in a period" in caplog.text
    assert model.identity()["text_prompt"] == prompt   # warned about, never rewritten


def test_a_well_formed_prompt_is_not_warned_about(caplog):
    with caplog.at_level("WARNING"):
        cf.groundedsam2(text_prompt="dragonfly.")
    assert "lowercase phrase" not in caplog.text


def test_no_detector_means_no_prompt_to_warn_about(caplog):
    with caplog.at_level("WARNING"):
        cf.sam2(text_prompt="Dragonfly")
    assert "lowercase phrase" not in caplog.text


# ---------------------------------------------------------------------------
# What a detection reports about the other boxes it found
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a, b, expected", [
    ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),    # disjoint
    ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),      # identical
    ([0, 0, 10, 10], [0, 0, 5, 10], 0.5),       # one inside the other
])
def test_box_iou(a, b, expected):
    from critterframe.segmentation.groundedsam import _box_iou
    assert _box_iou(a, b) == pytest.approx(expected)


def _detecting_model(monkeypatch, boxes):
    model = cf.groundedsam2(text_prompt="dragonfly.")
    monkeypatch.setattr(model, "detect_boxes", lambda image: boxes)
    monkeypatch.setattr(model, "_predict", lambda image, **kwargs:
                        (np.ones((20, 30), bool), 0.9))
    return model


def test_a_second_separate_box_is_reported(monkeypatch):
    model = _detecting_model(monkeypatch, [([0, 0, 10, 10], 0.8),
                                           ([20, 0, 30, 10], 0.4)])
    _mask, _score, info = model.predict(np.zeros((20, 30, 3), np.uint8))
    assert info["n_boxes"] == 2
    assert info["second_box_score"] == pytest.approx(0.4)
    assert info["second_box_iou"] == 0.0
    assert info["box_score"] == pytest.approx(0.8)


def test_a_single_box_reports_no_runner_up_without_a_missing_score(monkeypatch):
    """0.0 rather than missing: a missing value never passes an export filter."""
    model = _detecting_model(monkeypatch, [([0, 0, 10, 10], 0.8)])
    _mask, _score, info = model.predict(np.zeros((20, 30, 3), np.uint8))
    assert (info["n_boxes"], info["second_box_score"], info["second_box_iou"]) == (1, 0.0, None)


def test_detect_still_returns_the_best_box_alone(monkeypatch):
    model = cf.groundedsam2(text_prompt="dragonfly.")
    monkeypatch.setattr(model, "detect_boxes", lambda image: [([0, 0, 1, 1], 0.7)])
    assert model.detect(None) == ([0, 0, 1, 1], 0.7)
    monkeypatch.setattr(model, "detect_boxes", lambda image: [])
    assert model.detect(None) == (None, 0.0)
