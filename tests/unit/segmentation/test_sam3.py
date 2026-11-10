"""
The SAM3 segmenter's identity and decisions -- tested without torch, as for GroundedSAM2.

`import critterframe` must work without the `[torch]` extra, so every torch and
transformers import in `segmentation/sam3.py` is deferred inside a method. What is
worth testing without weights is what reaches the recipe hash (`identity()`) and what
`predict()` decides from the instances SAM3 hands back: which one is kept, what is
reported about the rest, and that finding none is a failure rather than a fallback.
Running the real checkpoint is two tests marked `gpu`, deselected by default.
"""

import sys
import types

import numpy as np
import pytest

import critterframe as cf
from critterframe.core.recipes import Recipe
from critterframe.segmentation.sam3 import SAM3, logit_to_probability

# ---------------------------------------------------------------------------
# Construction without torch
# ---------------------------------------------------------------------------


def test_constructing_the_model_loads_no_weights():
    model = SAM3()
    assert model.model is None
    assert model.tracker is None


def test_constructing_the_model_imports_no_torch():
    for name in ("torch", "transformers"):
        sys.modules.pop(name, None)

    cf.sam3()
    cf.sam3_points()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_a_recipe_naming_it_can_be_hashed_without_torch():
    recipe = Recipe("segment", "organisms", [cf.segment(cf.sam3())], part="organism")
    assert len(recipe.hash) == 16


def test_it_never_hashes_like_groundedsam2():
    """Different weights and a different method: never the same work, whatever the settings."""
    assert cf.sam3_points().identity() != cf.sam2().identity()
    one = Recipe("segment", "organisms", [cf.segment(cf.sam3())], part="organism")
    other = Recipe("segment", "organisms", [cf.segment(cf.groundedsam2())], part="organism")
    assert one.hash != other.hash


# ---------------------------------------------------------------------------
# _load(), with a faked transformers so no real weights are needed
# ---------------------------------------------------------------------------


class _FakeWeights:
    def to(self, device):
        return self


def _fake_transformers(model_from_pretrained):
    """A minimal fake `transformers`: the concept model's from_pretrained is the caller's."""

    class FakeProcessor:
        @staticmethod
        def from_pretrained(name):
            return object()

    class FakeModel:
        from_pretrained = staticmethod(model_from_pretrained)

    class FakeTracker:
        @staticmethod
        def from_pretrained(name):
            return _FakeWeights()

    return types.SimpleNamespace(
        Sam3Model=FakeModel,
        Sam3Processor=FakeProcessor,
        Sam3TrackerModel=FakeTracker,
        Sam3TrackerProcessor=FakeProcessor,
    )


def test_a_failed_load_is_retried_not_skipped_forever(monkeypatch):
    attempts = {"n": 0}

    def model_from_pretrained(name):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("the paging file is too small")
        return _FakeWeights()

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(model_from_pretrained))
    model = SAM3(device="cpu")

    with pytest.raises(RuntimeError, match="paging file"):
        model._load()
    assert model.model is None and model.processor is None  # nothing half-loaded left behind

    model._load()
    assert model.model is not None and model.processor is not None
    assert attempts["n"] == 2


def _gated_refusal(status):
    """The error transformers raises for a gated checkpoint: an OSError caused by the hub's own."""
    httpx = pytest.importorskip("httpx")
    errors = pytest.importorskip("huggingface_hub.errors")

    def from_pretrained(name):
        response = httpx.Response(status, request=httpx.Request("GET", f"https://huggingface.co/{name}"))
        try:
            raise errors.GatedRepoError("Cannot access gated repo", response=response)
        except errors.GatedRepoError as err:
            raise OSError("You are trying to access a gated repo.") from err

    return from_pretrained


@pytest.mark.parametrize(
    "status, says",
    [(401, "hf auth login"), (403, "has not been granted access")],
)
def test_a_gated_checkpoint_says_which_step_is_missing(monkeypatch, status, says):
    """401 is no login sent; 403 is logged in without access. Each needs a different fix."""
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(_gated_refusal(status)))
    model = SAM3(device="cpu")

    with pytest.raises(RuntimeError, match=says) as raised:
        model._load()
    assert "https://huggingface.co/facebook/sam3" in str(raised.value)
    assert model.model is None and model.processor is None


def test_any_other_load_failure_passes_through_untouched(monkeypatch):
    def from_pretrained(name):
        raise OSError("disk full")

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(from_pretrained))
    with pytest.raises(OSError, match="disk full"):
        SAM3(device="cpu")._load()


def test_each_path_loads_only_its_own_half(monkeypatch):
    """The concept model and the tracker are separate weights; neither path pays for the other."""
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(lambda name: _FakeWeights()))

    text = SAM3(device="cpu")
    text._load()
    assert text.model is not None and text.tracker is None

    points = SAM3(detect_bounds=False, device="cpu")
    points._load()
    assert points.tracker is not None and points.model is None


# ---------------------------------------------------------------------------
# identity()
# ---------------------------------------------------------------------------


def test_the_checkpoint_is_the_important_part():
    assert cf.sam3(model_name="a").identity() != cf.sam3(model_name="b").identity()


def test_the_prompting_strategy_is_in_the_identity_too():
    text = cf.sam3(detect_bounds=True).identity()
    points = cf.sam3(detect_bounds=False).identity()

    assert text != points
    assert "text_prompt" in text and "use_center_point" not in text
    assert "use_center_point" in points and "text_prompt" not in points
    assert "score_threshold" not in points


@pytest.mark.parametrize("kwargs", [{"text_prompt": "moth"}, {"score_threshold": 0.3}])
def test_every_text_setting_changes_the_identity(kwargs):
    assert cf.sam3(**kwargs).identity() != cf.sam3().identity()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_center_point": True},
        {"use_corner_points": True},
        {"retry_without_center": True},
        {"min_area_frac": 0.5},
    ],
)
def test_every_point_setting_changes_it_on_the_other_branch(kwargs):
    assert cf.sam3_points(**kwargs).identity() != cf.sam3_points().identity()


def test_the_other_branchs_settings_dont_move_it():
    """A text prompt on a model that never reads one must not invalidate its masks."""
    assert cf.sam3_points(text_prompt="moth").identity() == cf.sam3_points().identity()
    assert cf.sam3(use_center_point=True).identity() == cf.sam3().identity()


def test_the_device_is_not_part_of_the_identity():
    assert cf.sam3(device="cpu").identity() == cf.sam3(device="cuda").identity()


def test_the_implementation_version_is_carried_explicitly():
    assert cf.sam3().identity()["version"] == "1"


def test_sam3_points_is_sam3_without_the_text_prompt():
    assert cf.sam3_points().identity() == cf.sam3(detect_bounds=False).identity()


def test_the_points_shorthand_can_still_be_told_to_detect():
    assert cf.sam3_points(detect_bounds=True).detect_bounds is True


# ---------------------------------------------------------------------------
# mask_threshold: a logit everywhere, converted for SAM3's post-processing
# ---------------------------------------------------------------------------


def test_a_neutral_logit_is_sam3s_own_default():
    assert logit_to_probability(0.0) == pytest.approx(0.5)


def test_the_conversion_keeps_the_direction():
    """Negative grows the mask (a lower cutoff), positive shrinks it, as for SAM2."""
    assert logit_to_probability(-2.0) < logit_to_probability(0.0) < logit_to_probability(2.0)


def test_predict_hands_the_threshold_through(monkeypatch):
    seen = {}

    def instances(image, mask_threshold=0.0):
        seen["mask_threshold"] = mask_threshold
        return [(np.ones((20, 30), bool), [0, 0, 10, 10], 0.9)]

    model = cf.sam3()
    monkeypatch.setattr(model, "instances", instances)
    model.predict(np.zeros((20, 30, 3), np.uint8), mask_threshold=-1.5)
    assert seen["mask_threshold"] == -1.5


# ---------------------------------------------------------------------------
# What predict() decides from SAM3's instances
# ---------------------------------------------------------------------------


def _model_finding(monkeypatch, found):
    model = cf.sam3(text_prompt="dragonfly")
    monkeypatch.setattr(model, "instances", lambda image, mask_threshold=0.0: found)
    return model


def _mask(value):
    return np.full((20, 30), value, bool)


def test_the_best_instance_is_kept_and_the_runner_up_reported(monkeypatch):
    best, other = _mask(True), _mask(False)
    model = _model_finding(monkeypatch, [(best, [0, 0, 10, 10], 0.8), (other, [20, 0, 30, 10], 0.4)])
    mask, score, info = model.predict(np.zeros((20, 30, 3), np.uint8))

    assert mask is best
    assert score == pytest.approx(0.8)
    assert info["prompt"] == "text"
    assert info["box"] == [0, 0, 10, 10]
    assert info["n_boxes"] == 2
    assert info["second_box_score"] == pytest.approx(0.4)
    assert info["second_box_iou"] == 0.0


def test_a_single_instance_reports_no_runner_up_without_a_missing_score(monkeypatch):
    """0.0 rather than missing: a missing value never passes an export filter."""
    model = _model_finding(monkeypatch, [(_mask(True), [0, 0, 10, 10], 0.8)])
    _mask_, _score, info = model.predict(np.zeros((20, 30, 3), np.uint8))
    assert (info["n_boxes"], info["second_box_score"], info["second_box_iou"]) == (1, 0.0, None)


def test_finding_nothing_fails_rather_than_falling_back(monkeypatch):
    """A fallback to points would store a mask its recipe no longer describes."""
    model = _model_finding(monkeypatch, [])
    with pytest.raises(ValueError, match="dragonfly"):
        model.predict(np.zeros((20, 30, 3), np.uint8))


def test_the_info_keys_match_groundedsam2s(monkeypatch):
    """So a filter on a second organism carries over when the segmenter is swapped."""
    sam3_model = _model_finding(monkeypatch, [(_mask(True), [0, 0, 10, 10], 0.8)])
    _m, _s, sam3_info = sam3_model.predict(np.zeros((20, 30, 3), np.uint8))

    dino = cf.groundedsam2(text_prompt="dragonfly.")
    monkeypatch.setattr(dino, "detect_boxes", lambda image: [([0, 0, 10, 10], 0.8)])
    monkeypatch.setattr(dino, "_predict", lambda image, **kwargs: (_mask(True), 0.9))
    _m, _s, dino_info = dino.predict(np.zeros((20, 30, 3), np.uint8))

    assert set(dino_info) - {"prompt"} <= set(sam3_info)


def test_the_point_path_retries_without_the_centre_on_a_sliver(monkeypatch):
    calls = []

    def predict_points(image, points, labels, mask_threshold=0.0):
        calls.append(list(labels))
        sliver = np.zeros((100, 100), bool)
        sliver[0, 0] = len(calls) == 1
        return (sliver if len(calls) == 1 else np.ones((100, 100), bool)), 0.5

    model = cf.sam3_points(use_center_point=True, use_corner_points=True, retry_without_center=True)
    monkeypatch.setattr(model, "_predict_points", predict_points)
    mask, _score, info = model.predict(np.zeros((100, 100, 3), np.uint8))

    assert info["retried"] is True
    assert calls == [[1, 0, 0, 0, 0], [0, 0, 0, 0]]
    assert mask.all()


def test_the_point_geometry_is_groundedsam2s():
    """Shared, not copied, so the two point prompts can't drift apart."""
    from critterframe.segmentation import groundedsam, sam3

    assert sam3.prompt_points is groundedsam.prompt_points
    assert groundedsam.prompt_points(200, 100, True, True) == cf.sam2()._prompt_points(200, 100, True, True)


# ---------------------------------------------------------------------------
# Actually running the weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("factory", [cf.sam3, lambda: cf.sam3_points(use_center_point=True)])
def test_the_model_segments_a_real_image(draw_specimen, factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    image = draw_specimen(0)
    try:
        mask, score, info = factory().predict(image[..., ::-1])  # RGB, as segment() passes it
    except ValueError:
        pytest.skip("SAM3 found no organism in the synthetic drawing")

    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert 0 <= float(score) <= 1
    assert isinstance(info, dict)


@pytest.mark.gpu
def test_the_model_runs_through_a_real_run(image_project):
    pytest.importorskip("torch")

    result = cf.run_segments(image_project, steps=[cf.segment(cf.sam3_points())], limit=2, visualize=False)[
        "organism"
    ]
    assert result["processed"] + result["failed"] == 2
