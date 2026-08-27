"""
Spatial transforms, and the one thing they must never forget.

Every operation here moves pixels, and every one of them therefore has to pass
`applied=` to `Segment.replace()` so the mapping back to original coordinates
survives. A transform that forgets produces masks that look perfectly correct
beside their own cropped image and land on the wrong pixels of the parent -- with
nothing raising, and no way to tell from the stored mask which it was.

`test_recipes.py` tests the inversion through whole chains. This file tests each
operation's own arithmetic, plus the argument validation, which exists because
"crop(region='centre')" should fail immediately rather than at the end of a
four-hour run.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.transforms.crop import REGIONS
from helpers.synthetic import blob_mask

FRAME = (200, 300)          # height, width


def a_segment(mask=True):
    image = np.zeros((*FRAME, 3), np.uint8)
    image[60:100, 200:240] = 255
    return Segment(image, mask=blob_mask(FRAME) if mask else None,
                   occurrence_id="test")


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_every_named_region_crops_to_a_smaller_frame(region):
    cropped, info = cf.crop(region=region)(a_segment())
    assert cropped.shape[0] <= FRAME[0] and cropped.shape[1] <= FRAME[1]
    assert cropped.shape != FRAME
    # The box is reported in pixels of the frame it was applied to, which is
    # what a diagnostics dict has to say to be checkable against the image.
    assert (info["height"], info["width"]) == cropped.shape
    assert (info["source_height"], info["source_width"]) == FRAME


def test_a_named_region_is_fractional():
    """
    So one setting survives a change of camera resolution, which pixel
    coordinates would not.
    """
    small = Segment(np.zeros((100, 100, 3), np.uint8))
    large = Segment(np.zeros((400, 400, 3), np.uint8))

    assert cf.crop(region="upper_left")(small)[0].shape == (50, 50)
    assert cf.crop(region="upper_left")(large)[0].shape == (200, 200)


def test_an_explicit_box_is_in_pixels():
    cropped, _info = cf.crop(x=10, y=20, width=30, height=40)(a_segment())
    assert cropped.shape == (40, 30)


def test_a_crop_moves_the_mask_with_the_image():
    cropped, _info = cf.crop(region="upper_right")(a_segment())
    assert cropped.mask.shape == cropped.shape


@pytest.mark.parametrize("kwargs, message", [
    ({}, "needs region="),
    ({"x": 1}, "needs region="),
    ({"region": "center", "x": 1}, "either region="),
    ({"region": "centre"}, "unknown region"),
])
def test_bad_crop_arguments_fail_at_configuration_time(kwargs, message):
    """
    Not four hours into a run. An operation is configured once and executed
    thousands of times, so its arguments are checked where they are written.
    """
    with pytest.raises(ValueError, match=message):
        cf.crop(**kwargs)


# ---------------------------------------------------------------------------
# crop_to_mask
# ---------------------------------------------------------------------------


def test_cropping_to_the_mask_keeps_the_whole_organism():
    cropped, _info = cf.crop_to_mask(pad=0)(a_segment())
    assert cropped.mask.sum() == a_segment().mask.sum()


def test_padding_makes_the_frame_bigger_and_keeps_the_mask_the_same():
    tight, _info = cf.crop_to_mask(pad=0)(a_segment())
    padded, _info = cf.crop_to_mask(pad=0.5)(a_segment())

    assert padded.shape[0] > tight.shape[0]
    assert padded.mask.sum() == tight.mask.sum()


def test_cropping_to_the_mask_needs_a_mask():
    with pytest.raises(ValueError, match="has no mask yet"):
        cf.crop_to_mask()(a_segment(mask=False))


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


def test_rotating_expands_the_canvas_so_nothing_is_clipped():
    rotated, _info = cf.rotate(45)(a_segment())
    assert rotated.shape[0] > FRAME[0]
    assert rotated.mask.sum() > 0


def test_a_right_angle_swaps_the_axes():
    """
    Within a pixel: the expanded canvas is computed from the rotated corners
    and rounds outward, which is the right way to round when the alternative is
    clipping the specimen.
    """
    rotated, _info = cf.rotate(90)(a_segment())
    assert abs(rotated.shape[0] - FRAME[1]) <= 1
    assert abs(rotated.shape[1] - FRAME[0]) <= 1


def test_rotating_by_nothing_changes_nothing():
    rotated, _info = cf.rotate(0)(a_segment())
    assert rotated.shape == FRAME
    assert np.array_equal(rotated.mask, a_segment().mask)


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_scaling_multiplies_both_dimensions():
    resized, _info = cf.resize(scale=0.5)(a_segment())
    assert resized.shape == (100, 150)


def test_one_dimension_given_keeps_the_aspect_ratio():
    """
    Distorting an organism's proportions would corrupt every shape trait
    measured afterward.
    """
    resized, _info = cf.resize(width=150)(a_segment())
    assert resized.shape == (100, 150)


def test_both_dimensions_given_are_obeyed():
    resized, _info = cf.resize(width=100, height=100)(a_segment())
    assert resized.shape == (100, 100)


@pytest.mark.parametrize("kwargs, message", [
    ({}, "needs scale="),
    ({"scale": 2, "width": 10}, "either scale="),
])
def test_bad_resize_arguments_fail_at_configuration_time(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cf.resize(**kwargs)


def test_a_resized_mask_stays_boolean():
    """
    Interpolating a boolean mask as a float and thresholding it later would
    quietly change its area; the operation is responsible for its own dtype.
    """
    resized, _info = cf.resize(scale=2.0)(a_segment())
    assert resized.mask.dtype == bool


# ---------------------------------------------------------------------------
# remove_background
# ---------------------------------------------------------------------------


def test_the_background_is_blanked_and_the_organism_is_not():
    blanked, info = cf.remove_background()(a_segment())
    mask = a_segment().mask

    assert (blanked.image[~mask] == 0).all()
    assert np.array_equal(blanked.image[mask], a_segment().image[mask])
    assert 0 < info["background_fraction"] < 1


def test_the_fill_value_is_configurable():
    blanked, _info = cf.remove_background(fill=255)(a_segment())
    assert (blanked.image[~a_segment().mask] == 255).all()


def test_removing_the_background_moves_no_pixels():
    """
    Which is why it needs no `applied=`: the mapping back to original
    coordinates is untouched and the mask passes through unchanged.
    """
    original = a_segment()
    blanked, _info = cf.remove_background()(original)

    assert np.allclose(blanked.matrix, original.matrix)
    assert np.array_equal(blanked.mask, original.mask)


def test_removing_the_background_needs_a_mask():
    with pytest.raises(ValueError, match="has no mask yet"):
        cf.remove_background()(a_segment(mask=False))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.stages = []

    def collect(self, occurrence_id, stage, image):
        self.stages.append(stage)
        assert image.dtype == np.uint8, "panels must arrive display-ready"


@pytest.mark.parametrize("operation, stage", [
    (cf.crop(region="center"), "crop"),
    (cf.crop_to_mask(), "crop_to_mask"),
    (cf.rotate(15), "rotate"),
    (cf.resize(scale=0.5), "resize"),
    (cf.remove_background(), "remove_background"),
])
def test_each_operation_emits_a_display_ready_panel_named_for_the_step(
        operation, stage):
    """
    The stage names the column a panel lands in, so it is named for the STEP
    rather than for the occurrence.
    """
    sink = RecordingSink()
    image = np.zeros((*FRAME, 3), np.uint8)
    operation(Segment(image, mask=blob_mask(FRAME), occurrence_id="test",
                      panel_sink=sink))
    assert sink.stages == [stage]
