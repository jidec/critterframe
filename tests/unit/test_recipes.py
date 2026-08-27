"""
The three types everything is built from -- and the invariant that outranks all
the others.

A persisted mask is ALWAYS in the coordinates of the original analysis image,
whatever crops and rotations produced it. `Segment` carries the affine that maps
original coordinates to its own, transforms compose onto it, and
`mask_in_original_coordinates()` inverts the whole chain in one step before
persistence. When that breaks, nothing raises: the mask still looks correct next
to its cropped image, and it simply lands on the wrong pixels of the parent.
Half this file exists to make that failure loud.

The other half is recipe hashing, tested RELATIONALLY: a changed parameter must
move the hash, and a display flag must not. The three literal digests that catch
a change to the machinery itself live in test_hash_stability.py, which explains
why they are separate.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import (
    IDENTITY,
    Metric,
    Operation,
    Recipe,
    Segment,
    Segmentation,
    Transform,
    canonical_json,
    compose,
    describe,
    hash_spec,
    load_json,
    model_identity,
)
from helpers.models import ThresholdModel, UnidentifiedModel

FRAME = (200, 300)
BLOB = (slice(60, 100), slice(200, 240))     # y, x -- upper right of the frame


def a_frame():
    """A dark frame with one bright blob at a known place, and its true mask."""
    image = np.zeros((*FRAME, 3), np.uint8)
    image[BLOB] = 255
    truth = np.zeros(FRAME, bool)
    truth[BLOB] = True
    return image, truth


def iou(mask, other):
    union = (mask | other).sum()
    return float((mask & other).sum() / union) if union else 1.0


def a_segment(**kwargs):
    image, truth = a_frame()
    return Segment(image, mask=truth, occurrence_id="test", **kwargs)


# ---------------------------------------------------------------------------
# Segment basics
# ---------------------------------------------------------------------------


def test_a_mask_is_stored_as_booleans():
    """Callers hand in whatever their operation produced; `> 0` is the rule."""
    segment = Segment(np.zeros((4, 4, 3), np.uint8), mask=np.eye(4) * 7)
    assert segment.mask.dtype == bool
    assert segment.mask.sum() == 4


def test_a_new_segment_maps_original_coordinates_to_itself():
    assert np.allclose(a_segment().matrix, IDENTITY)
    assert a_segment().original_shape == FRAME


def test_shape_is_the_current_working_frame():
    assert a_segment().shape == FRAME


def test_rgb_swaps_channels_and_widens_grayscale():
    """
    Everything in the package works in OpenCV BGR; models want RGB. A grayscale
    image has to come back three-channel or a model receives the wrong rank.
    """
    image = np.zeros((2, 2, 3), np.uint8)
    image[..., 0] = 255                       # blue in BGR
    assert Segment(image).rgb[0, 0].tolist() == [0, 0, 255]
    assert Segment(np.zeros((2, 2), np.uint8)).rgb.shape == (2, 2, 3)


def test_require_mask_says_what_to_do_about_it():
    """
    A metric with no mask fails here with a useful message rather than three
    frames down with a NoneType error.
    """
    with pytest.raises(ValueError, match="put a segment"):
        Segment(np.zeros((4, 4, 3), np.uint8)).require_mask()


def test_replace_leaves_the_original_untouched():
    """
    Operations return new segments rather than mutating, so a recipe's
    intermediate states stay inspectable and a failed operation cannot leave a
    half-modified segment behind.
    """
    original = a_segment()
    replaced = original.replace(image=np.zeros((10, 10, 3), np.uint8))

    assert original.shape == FRAME
    assert replaced.shape == (10, 10)
    assert replaced.occurrence_id == original.occurrence_id


def test_replace_keeps_the_mask_unless_told_otherwise():
    segment = a_segment()
    assert segment.replace().mask is not None
    assert segment.replace(mask=False).mask is None


def test_for_part_relabels_without_redoing_the_work():
    """
    How a multi-output segmentation forks one shared, preprocessed segment into
    a branch per part.
    """
    segment = a_segment()
    wing = segment.for_part("wing")
    assert wing.part == "wing"
    assert segment.part == "organism"
    assert np.array_equal(wing.image, segment.image)


# ---------------------------------------------------------------------------
# The affine chain
# ---------------------------------------------------------------------------


def test_compose_applies_the_new_transform_after_the_old():
    """
    `existing` maps original -> current and `applied` maps current -> new.
    Getting the multiplication backwards produces masks that look plausible and
    land in the wrong place, which is the entire reason this is a named
    function.
    """
    shift_right = np.array([[1.0, 0, 10], [0, 1.0, 0]])
    shift_down = np.array([[1.0, 0, 0], [0, 1.0, 5]])
    assert np.allclose(compose(shift_right, shift_down),
                       [[1, 0, 10], [0, 1, 5]])


def test_a_transform_that_moves_pixels_composes_onto_the_matrix():
    moved, _info = cf.crop(region="upper_right")(a_segment())
    assert not np.allclose(moved.matrix, IDENTITY)


def test_original_shape_survives_the_chain():
    """It is the canvas size the inversion has to paint back onto."""
    moved, _info = cf.crop(region="upper_right")(a_segment())
    rotated, _info = cf.rotate(30)(moved)
    assert rotated.original_shape == FRAME


# ---------------------------------------------------------------------------
# mask_in_original_coordinates -- the invariant
# ---------------------------------------------------------------------------


CHAINS = {
    "no transforms": [],
    "crop": [cf.crop(region="upper_right")],
    "crop then rotate": [cf.crop(region="upper_right"), cf.rotate(30)],
    "crop rotate resize": [cf.crop(region="upper_right"), cf.rotate(30),
                           cf.resize(scale=2.0)],
    "crop_to_mask": [cf.crop_to_mask()],
    "resize alone": [cf.resize(scale=0.5)],
    "rotate alone": [cf.rotate(90)],
}


@pytest.mark.parametrize("chain", list(CHAINS), ids=list(CHAINS))
def test_a_mask_returns_to_where_the_organism_actually_is(chain):
    """
    Whatever a recipe does to the frame, the mask must come back pointing at
    the same pixels of the ORIGINAL image, at the original size.

    IoU below 1.0 is expected wherever a rotation or resize resampled the mask
    -- interpolation is lossy, the coordinate frame is not -- so the threshold
    is generous while the centroid check is tight. A wrong composition moves
    the blob by tens of pixels, not by one.
    """
    image, truth = a_frame()
    segment = Segment(image, mask=truth, occurrence_id="test")
    for operation in CHAINS[chain]:
        segment, _info = operation(segment)

    restored = segment.mask_in_original_coordinates()
    ys, xs = np.nonzero(restored)

    assert restored.shape == FRAME
    assert iou(restored, truth) > 0.8
    assert abs(xs.mean() - 220) < 3
    assert abs(ys.mean() - 80) < 3


def test_an_untransformed_mask_comes_back_unchanged():
    """The fast path: no warp, no resampling, exactly the pixels handed in."""
    _image, truth = a_frame()
    assert np.array_equal(a_segment().mask_in_original_coordinates(), truth)


def test_a_crop_at_the_origin_is_still_restored_to_full_size():
    """
    The trap the shape check exists for. An upper-LEFT crop translates by
    (0, 0), so its affine IS the identity while its canvas is smaller.
    Returning early on the matrix alone would persist a mask sized to the crop,
    which then silently disagrees with every other mask of the same occurrence.
    """
    image, truth = a_frame()
    segment = Segment(image, mask=truth, occurrence_id="test")
    cropped, _info = cf.crop(region="upper_left")(segment)

    assert np.allclose(cropped.matrix, IDENTITY)
    assert cropped.shape != FRAME
    assert cropped.mask_in_original_coordinates().shape == FRAME


def test_inverting_without_a_mask_raises():
    with pytest.raises(ValueError, match="has no mask yet"):
        Segment(np.zeros((*FRAME, 3), np.uint8)).mask_in_original_coordinates()


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.collected = []

    def collect(self, occurrence_id, stage, image):
        self.collected.append((occurrence_id, stage, image))


def test_emitting_a_panel_without_a_sink_does_nothing():
    """
    Operations call emit_panel unconditionally; it no-ops for every occurrence
    outside the sample, which is the great majority of them.
    """
    a_segment().emit_panel(np.zeros((4, 4, 3), np.uint8), "stage")


def test_a_sink_receives_the_stage_and_the_occurrence():
    sink = RecordingSink()
    segment = a_segment(panel_sink=sink)
    segment.emit_panel(np.zeros((4, 4, 3), np.uint8), "orientation")

    assert [(occurrence, stage) for occurrence, stage, _ in sink.collected] == [
        ("test", "orientation")]


def test_the_sink_survives_replace():
    """
    Otherwise a transform chain would emit panels for its first step and go
    quiet for the rest.
    """
    sink = RecordingSink()
    assert a_segment(panel_sink=sink).replace().panel_sink is sink


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def test_an_operations_spec_covers_what_changes_its_output():
    operation = Operation("thing", lambda segment: segment,
                          parameters={"a": 1}, version="2")
    assert operation.spec() == {"name": "thing", "kind": "operation",
                                "version": "2", "parameters": {"a": 1}}


def test_a_model_reaches_the_spec_through_its_own_identity():
    operation = Operation("segment", lambda segment, model: segment,
                          model=ThresholdModel(cutoff=120))
    assert operation.spec()["model"] == {"class": "ThresholdModel",
                                         "cutoff": 120, "erode": 0}


def test_a_model_without_an_identity_is_only_its_class_name():
    """
    The honest answer for an object that never said which weights it holds --
    and the reason register_model() exists: two fine-tunes of one class hash
    alike here.
    """
    assert model_identity(UnidentifiedModel()) == {"class": "UnidentifiedModel"}


def test_invoke_passes_parameters_and_keeps_the_model_out_of_them():
    """
    Parameters must be JSON-serializable to be hashable, and a loaded network
    isn't. It arrives as `model=` instead.
    """
    seen = {}

    def function(segment, model=None, **parameters):
        seen.update(parameters, model=model)
        return segment

    model = ThresholdModel()
    Operation("thing", function, parameters={"a": 1}, model=model).invoke(None)
    assert seen == {"a": 1, "model": model}


def test_the_three_kinds_are_distinguishable():
    assert Transform("t", None).kind == "transform"
    assert Segmentation("s", None).kind == "segment"
    assert Metric("m", None).kind == "metric"


def test_a_metric_carries_its_unit_and_stored_name():
    """
    `unit` is inside the recipe hash, which is what makes measuring in
    millimetres a different recipe -- so a re-calibration can never invalidate
    a stored trait.
    """
    metric = Metric("mask_area", None, unit="px2", metric_name="area_px")
    assert metric.spec()["unit"] == "px2"
    assert metric.spec()["metric_name"] == "area_px"


def test_a_metric_defaults_to_being_stored_under_its_own_name():
    assert Metric("body_length", None).metric_name == "body_length"


def test_prepare_is_optional():
    assert Operation("thing", None).prepare(context=None) is None


# ---------------------------------------------------------------------------
# Recipe identity -- sensitivity
# ---------------------------------------------------------------------------


def base_recipe(**kwargs):
    settings = dict(kind="metric", name="traits", part="organism")
    settings.update(kwargs)
    operations = settings.pop("operations", None) or [cf.body_length()]
    return Recipe(operations=operations, **settings)


def test_the_same_configuration_hashes_the_same():
    """
    Two recipes hash alike exactly when running them would do the same work.
    Built separately, in different objects, with no shared state.
    """
    assert base_recipe().hash == base_recipe().hash


@pytest.mark.parametrize("difference", [
    {"name": "other_run"},
    {"part": "wing"},
    {"from_part": "organism"},
    {"inputs": {"masks": "reference"}},
    {"kind": "segment"},
    {"operations": [cf.max_width()]},
    {"operations": [cf.body_length(), cf.max_width()]},
])
def test_a_different_recipe_hashes_differently(difference):
    """
    The sensitivity family. Miss one of these and the package serves cached
    work from a recipe that no longer describes it.
    """
    assert base_recipe(**difference).hash != base_recipe().hash


def test_operation_order_is_part_of_identity():
    """
    remove_appendages() then orient() is not the same processing as orient()
    then remove_appendages(), and the results genuinely differ.
    """
    forward = base_recipe(operations=[cf.remove_appendages(), cf.orient()])
    backward = base_recipe(operations=[cf.orient(), cf.remove_appendages()])
    assert forward.hash != backward.hash


def test_a_changed_operation_parameter_moves_the_hash():
    assert (base_recipe(operations=[cf.remove_appendages(relative_radius=0.1)]).hash
            != base_recipe(operations=[cf.remove_appendages(relative_radius=0.2)]).hash)


def test_a_bumped_version_moves_the_hash():
    """What a maintainer bumps by hand when an implementation's output changes."""
    old = Operation("thing", None, version="1")
    new = Operation("thing", None, version="2")
    assert base_recipe(operations=[old]).hash != base_recipe(operations=[new]).hash


def test_a_different_checkpoint_moves_the_hash():
    """Two runs with different weights are never mistaken for equivalent work."""
    first = base_recipe(kind="segment",
                        operations=[cf.segment(ThresholdModel(cutoff=100))])
    second = base_recipe(kind="segment",
                         operations=[cf.segment(ThresholdModel(cutoff=120))])
    assert first.hash != second.hash


def test_a_renamed_metric_moves_the_hash():
    """
    Storing the same measurement under a second name is different work -- the
    export gets another column -- so it must not be skipped as already done.
    """
    assert (base_recipe(operations=[cf.mask_area(name="area_px")]).hash
            != base_recipe(operations=[cf.mask_area()]).hash)


# ---------------------------------------------------------------------------
# Recipe identity -- stability
# ---------------------------------------------------------------------------


def test_parameter_order_does_not_move_the_hash():
    """
    `canonical_json` sorts keys precisely so a dict written in a different
    order is the same configuration.
    """
    forward = Operation("thing", None, parameters={"a": 1, "b": 2})
    backward = Operation("thing", None, parameters={"b": 2, "a": 1})
    assert forward.spec() == backward.spec()
    assert hash_spec(forward.spec()) == hash_spec(backward.spec())


def test_a_numpy_number_hashes_as_its_python_value():
    """
    A parameter that arrived from an array computation must not be different
    work from the same number typed by hand.
    """
    plain = Operation("thing", None, parameters={"scale": 2.0})
    numpy = Operation("thing", None, parameters={"scale": np.float64(2.0)})
    assert hash_spec(plain.spec()) == hash_spec(numpy.spec())


def test_a_panel_sink_is_not_part_of_a_segments_identity():
    """
    Nothing about visualization reaches a hash: the same recipe with diagnostics
    on produces identical results, so a rerun must recognize the cached work.
    """
    assert base_recipe().hash == base_recipe().hash


def test_describe_carries_the_spec_as_well_as_the_hash():
    """
    Stored in full on the run so it stays readable years later, even if the
    operation that produced it has since been renamed or deleted.
    """
    recipe = base_recipe()
    described = describe(recipe)
    assert described["recipe_hash"] == recipe.hash
    assert described["recipe"] == recipe.spec()


def test_operations_of_selects_one_kind_in_order():
    recipe = base_recipe(operations=[cf.remove_appendages(), cf.orient(),
                                     cf.body_length()])
    assert [op.name for op in recipe.operations_of("transform")] == [
        "remove_appendages", "orient"]
    assert [op.name for op in recipe.operations_of("metric")] == ["body_length"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_canonical_json_is_deterministic_and_compact():
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


@pytest.mark.parametrize("value, expected", [
    (np.float64(1.5), "1.5"),
    (np.int64(3), "3"),
    (np.array([1, 2]), "[1,2]"),
    (np.bool_(True), "true"),
])
def test_numpy_values_serialize_as_plain_json(value, expected):
    """
    Metric values arrive from numpy constantly. Without this they would not be
    storable at all, since a value that can't be serialized can't be recorded.
    """
    assert canonical_json(value) == expected


def test_an_unserializable_value_raises():
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json({"model": object()})


def test_load_json_passes_none_through():
    """A null column is a value that was never set, not the string "null"."""
    assert load_json(None) is None
    assert load_json('{"a":1}') == {"a": 1}


def test_a_hash_is_short_and_stable():
    assert len(hash_spec({"a": 1})) == 16
    assert hash_spec({"a": 1}) == hash_spec({"a": 1})
