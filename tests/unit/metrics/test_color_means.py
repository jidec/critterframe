"""
Mean colour and lightness, measured over the organism and nothing else.

The values are relative, not colorimetric: comparable within a project shot on one rig, and comparable across rigs
only once colour calibration exists.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment
from helpers.synthetic import flat


# ---------------------------------------------------------------------------
# mean_lightness / mean_color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(0, 0.0), (255, 1.0)])
def test_lightness_runs_from_black_to_white(value, expected):
    assert cf.mean_lightness()(flat((value, value, value))) == pytest.approx(
        expected, abs=0.02)


def test_lightness_is_perceptual_rather_than_a_mean_pixel_value():
    """
    CIE L*, not the average of the channels. Mid-grey reads well above 0.5,
    which is correct and is the whole reason to use L* -- it tracks how light
    the organism LOOKS, which is what a trait like melanisation is about.
    """
    mid = cf.mean_lightness()(flat((128, 128, 128)))
    assert 0.5 < mid < 0.6
    assert (cf.mean_lightness()(flat((60, 60, 60)))
            < mid
            < cf.mean_lightness()(flat((200, 200, 200))))


def test_mean_colour_reports_three_channels_in_rgb_order():
    """
    RGB rather than the BGR the package works in, because the value is read by
    a person and by R, and "r" meaning blue would be a trap.
    """
    colour = cf.mean_color()(flat((0, 0, 255)))     # BGR red
    assert colour["r"] > 0.9
    assert colour["g"] < 0.1 and colour["b"] < 0.1


def test_mean_colour_is_a_fraction_per_channel():
    colour = cf.mean_color()(flat((128, 128, 128)))
    assert all(value == pytest.approx(0.5, abs=0.02) for value in colour.values())


def test_two_organisms_of_the_same_mean_can_look_nothing_alike():
    """
    Documented in the operation itself, and worth pinning: an all-grey specimen
    and a half-black half-white one have the same mean colour. It is why
    mean_color is paired with black_fraction or a colour cluster rather than
    used alone.
    """
    grey = cf.mean_color()(flat((128, 128, 128)))

    striped = np.zeros((60, 60, 3), np.uint8)
    striped[::2] = 255
    patterned = cf.mean_color()(Segment(striped, mask=np.ones((60, 60), bool)))

    assert all(abs(grey[key] - patterned[key]) < 0.05 for key in grey)


def test_a_grayscale_image_is_measurable():
    """
    A store can hand back a single-channel working view for a grayscale
    original, and a colour metric that assumed three channels would fail on the
    one kind of image where its answer is least surprising.
    """
    gray = np.full((40, 40), 200, np.uint8)
    segment = Segment(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                      mask=np.ones((40, 40), bool))
    assert cf.mean_lightness()(segment) == pytest.approx(
        cf.mean_lightness()(flat((200, 200, 200))))


# ---------------------------------------------------------------------------
# white_balanced_color / background_color
# ---------------------------------------------------------------------------


def organism_on_ground(organism, ground, shape=(60, 60)):
    """A square organism of one colour centred on a ground of another."""
    image = np.full((*shape, 3), ground, np.uint8)
    mask = np.zeros(shape, bool)
    mask[20:40, 20:40] = True
    image[mask] = organism
    return Segment(image, mask=mask, occurrence_id="test")


def test_white_balance_leaves_an_already_grey_frame_alone():
    segment = organism_on_ground((90, 90, 90), (170, 170, 170))
    assert cf.white_balanced_color()(segment) == pytest.approx(
        cf.mean_color()(segment), abs=0.01)


def test_white_balance_removes_a_whole_frame_colour_cast():
    """
    A per-channel gain over the whole frame, organism included, is exactly
    what grey-world undoes: the balanced organism reads as grey again.
    """
    # Grey organism on grey ground, blue channel gained 1.5x.
    segment = organism_on_ground((150, 100, 100), (240, 160, 160))
    balanced = cf.white_balanced_color()(segment)
    assert balanced["b"] == pytest.approx(balanced["r"], abs=0.02)
    assert cf.mean_color()(segment)["b"] > cf.mean_color()(segment)["r"] + 0.15


def test_background_is_measured_outside_the_mask():
    background = cf.background_color()(organism_on_ground((255, 255, 255), (0, 0, 255)))
    assert background["r"] > 0.9
    assert background["g"] < 0.1 and background["b"] < 0.1


def test_contrast_is_signed_by_which_is_lighter():
    light_on_dark = cf.background_color()(organism_on_ground((230, 230, 230), (30, 30, 30)))
    dark_on_light = cf.background_color()(organism_on_ground((30, 30, 30), (230, 230, 230)))
    assert light_on_dark["contrast"] > 0.5
    assert dark_on_light["contrast"] < -0.5


def test_a_mask_covering_the_whole_frame_has_no_background():
    with pytest.raises(ValueError, match="no background"):
        cf.background_color()(flat((128, 128, 128)))
