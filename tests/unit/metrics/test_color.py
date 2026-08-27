"""
Colour traits, measured over the organism and nothing else.

The shared rule is in `_masked_pixels`: every colour metric reads the pixels
inside the mask. That is not an optimization -- a mean colour taken over the
whole frame is mostly a measurement of the substrate the specimen was
photographed on, and it would vary with the background across a collection while
looking exactly like a biological signal.

The values themselves are relative, not colorimetric: "how much of this organism
is black" is comparable within a project shot on one rig, and comparable across
rigs only once colour calibration exists.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment


def half_and_half(top=(0, 0, 0), bottom=(255, 255, 255), shape=(100, 100)):
    """An image split top/bottom, with a mask covering only the top half."""
    image = np.zeros((*shape, 3), np.uint8)
    image[:shape[0] // 2] = top
    image[shape[0] // 2:] = bottom

    mask = np.zeros(shape, bool)
    mask[:shape[0] // 2] = True
    return Segment(image, mask=mask, occurrence_id="test")


def flat(colour, shape=(60, 60)):
    """A uniformly coloured organism filling the whole frame."""
    image = np.full((*shape, 3), colour, np.uint8)
    return Segment(image, mask=np.ones(shape, bool), occurrence_id="test")


# ---------------------------------------------------------------------------
# Only the organism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction()])
def test_the_background_never_reaches_the_measurement(metric):
    """
    The rule every colour metric shares. Change what is OUTSIDE the mask and
    nothing may move -- otherwise the trait is partly a measurement of the leaf
    the animal was sitting on.
    """
    dark_background = half_and_half(top=(30, 30, 30), bottom=(0, 0, 0))
    light_background = half_and_half(top=(30, 30, 30), bottom=(255, 255, 255))
    assert metric(dark_background) == metric(light_background)


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


# ---------------------------------------------------------------------------
# black_fraction
# ---------------------------------------------------------------------------


def test_a_black_organism_is_all_black():
    assert cf.black_fraction()(flat((0, 0, 0))) == 1.0


def test_a_white_organism_is_none():
    assert cf.black_fraction()(flat((255, 255, 255))) == 0.0


def test_half_dark_is_half():
    image = np.zeros((100, 100, 3), np.uint8)
    image[50:] = 255
    segment = Segment(image, mask=np.ones((100, 100), bool))
    assert cf.black_fraction()(segment) == pytest.approx(0.5)


def test_the_threshold_is_configurable_and_hashed():
    """
    Where "dark" falls is a judgement about the imaging, so it is a parameter
    -- and being a parameter it is in the recipe hash, so changing it is new
    work rather than a silently different number under the same name.
    """
    grey = flat((100, 100, 100))                # about 0.43 lightness
    assert cf.black_fraction(threshold=0.2)(grey) == 0.0
    assert cf.black_fraction(threshold=0.8)(grey) == 1.0
    assert (cf.black_fraction(threshold=0.2).spec()
            != cf.black_fraction(threshold=0.8).spec())


# ---------------------------------------------------------------------------
# hue_fraction
# ---------------------------------------------------------------------------


def test_a_red_organism_reads_as_red_and_not_as_yellow():
    red = flat((0, 0, 255))
    assert cf.red_fraction()(red) > 0.9
    assert cf.yellow_fraction()(red) < 0.1


def test_a_yellow_organism_reads_as_yellow():
    assert cf.yellow_fraction()(flat((0, 255, 255))) > 0.9


def test_a_grey_organism_is_no_hue_at_all():
    """
    The saturation floor. Without it, sensor noise in a grey specimen would be
    assigned a hue and every neutral moth would report a little red.
    """
    grey = flat((128, 128, 128))
    assert cf.red_fraction()(grey) == 0.0
    assert cf.yellow_fraction()(grey) == 0.0


def test_a_very_dark_pixel_is_no_hue_either():
    """
    The value floor, for the same reason: hue is meaningless in shadow.
    """
    assert cf.red_fraction()(flat((0, 0, 20))) == 0.0


def test_each_hue_is_stored_under_its_own_name():
    """
    Two configurations of one operation in one recipe, which is exactly what
    metric_name exists for.
    """
    assert cf.red_fraction().metric_name == "red_fraction"
    assert cf.yellow_fraction().metric_name == "yellow_fraction"
    assert cf.red_fraction().spec() != cf.yellow_fraction().spec()


def test_an_unknown_hue_name_raises():
    with pytest.raises((KeyError, ValueError)):
        cf.hue_fraction("chartreuse")(flat((0, 255, 255)))


# ---------------------------------------------------------------------------
# Shared contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction()])
def test_every_colour_metric_needs_a_mask(metric):
    with pytest.raises(ValueError, match="has no mask yet"):
        metric(Segment(np.zeros((10, 10, 3), np.uint8)))


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction()])
def test_every_colour_metric_reports_a_fraction(metric):
    assert metric.unit == "fraction"


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
