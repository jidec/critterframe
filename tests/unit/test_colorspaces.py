"""
Colour space conversion in canonical units.

Checked against values worked out independently (pure red is Lab 53.2 / 80.1 / 67.2, mid-grey is 0.2158 in linear
light), and by round trip. The one detail the whole module exists for is that Lab comes out in its TRUE range: OpenCV's
8-bit path stores a/b offset by +128 and hue on 0-179, and a chroma or a hue arc computed from those is wrong.
"""

import numpy as np
import pytest

from critterframe.colorspaces import (
    SPACES,
    convert,
    get_space,
    in_arc,
    normalize,
    to_bgr,
)

RED = (0, 0, 255)       # BGR
WHITE = (255, 255, 255)
GREY = (128, 128, 128)


def px(*colours):
    return np.array(colours, dtype=np.uint8)


def sweep():
    """A spread of colours across the gamut, saturated and not."""
    rng = np.random.default_rng(0)
    random = rng.integers(0, 256, (200, 3), dtype=np.uint8)
    corners = np.array([(b, g, r) for b in (0, 255) for g in (0, 255) for r in (0, 255)], np.uint8)
    return np.vstack([random, corners])


# ---------------------------------------------------------------------------
# convert: known values
# ---------------------------------------------------------------------------


def test_rgb_reorders_bgr_and_scales_to_a_fraction():
    assert convert(px((10, 20, 255)), "rgb")[0] == pytest.approx([1.0, 20 / 255, 10 / 255])


def test_hsv_of_pure_red_is_hue_zero_full_saturation_and_value():
    assert convert(px(RED), "hsv")[0] == pytest.approx([0.0, 1.0, 1.0])


def test_hue_is_in_degrees_not_opencvs_half_degrees():
    green = convert(px((0, 255, 0)), "hsv")[0]
    assert green[0] == pytest.approx(120.0)


def test_hls_of_pure_red():
    assert convert(px(RED), "hls")[0] == pytest.approx([0.0, 0.5, 1.0])


def test_lab_of_pure_red():
    assert convert(px(RED), "lab")[0] == pytest.approx([53.2, 80.1, 67.2], abs=0.3)


def test_lab_is_true_range_not_offset():
    """Grey must land at a=b=0, not OpenCV's 8-bit-storage +128."""
    lab = convert(px(GREY), "lab")[0]
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_lab_lightness_runs_zero_to_one_hundred():
    assert convert(px((0, 0, 0), WHITE), "lab")[:, 0] == pytest.approx([0.0, 100.0], abs=0.1)


def test_lch_is_lab_in_polar_form():
    lab = convert(px(RED), "lab")[0]
    lch = convert(px(RED), "lch")[0]
    assert lch[0] == pytest.approx(lab[0])
    assert lch[1] == pytest.approx(np.hypot(lab[1], lab[2]))
    assert lch[2] == pytest.approx(np.degrees(np.arctan2(lab[2], lab[1])), abs=1e-3)
    assert lch[2] == pytest.approx(40.0, abs=0.5)


def test_grey_has_no_chroma():
    assert convert(px(GREY), "lch")[0, 1] == pytest.approx(0.0, abs=0.5)


def test_hue_is_never_negative():
    """arctan2 is signed; a hue below zero would fall outside every arc."""
    hues = convert(sweep(), "lch")[:, 2]
    assert hues.min() >= 0.0 and hues.max() < 360.0


def test_linear_rgb_removes_the_gamma_curve():
    assert convert(px(GREY), "linrgb")[0] == pytest.approx([0.2158] * 3, abs=1e-3)


def test_linear_rgb_keeps_the_endpoints_and_the_toe():
    linear = convert(px((0, 0, 0), WHITE, (10, 10, 10)), "linrgb")
    assert linear[0] == pytest.approx([0, 0, 0])
    assert linear[1] == pytest.approx([1, 1, 1])
    assert linear[2] == pytest.approx([10 / 255 / 12.92] * 3)      # below the 0.04045 knee, the curve is linear


# ---------------------------------------------------------------------------
# convert: shape and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space", sorted(SPACES))
def test_shape_is_preserved_and_the_result_is_float32(space):
    image = np.zeros((7, 5, 3), np.uint8)
    for pixels in (image, image.reshape(-1, 3)):
        converted = convert(pixels, space)
        assert converted.shape == pixels.shape
        assert converted.dtype == np.float32


def test_a_whole_image_converts_like_its_pixel_list():
    image = np.random.default_rng(1).integers(0, 256, (6, 4, 3), dtype=np.uint8)
    assert np.array_equal(convert(image, "lab").reshape(-1, 3), convert(image.reshape(-1, 3), "lab"))


@pytest.mark.parametrize("space", sorted(SPACES))
def test_no_pixels_is_no_values(space):
    assert convert(np.zeros((0, 3), np.uint8), space).shape == (0, 3)


def test_only_uint8_is_accepted():
    with pytest.raises(TypeError, match="uint8"):
        convert(np.zeros((4, 3), np.float32), "lab")


def test_the_last_axis_must_be_three():
    with pytest.raises(ValueError, match="last axis"):
        convert(np.zeros((4, 4), np.uint8), "lab")


def test_an_unknown_space_names_the_valid_ones():
    with pytest.raises(ValueError, match="hsv"):
        convert(px(RED), "cmyk")


# ---------------------------------------------------------------------------
# to_bgr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space", sorted(SPACES))
def test_converting_and_coming_back_recovers_the_pixels(space):
    pixels = sweep()
    back = to_bgr(convert(pixels, space), space)
    assert back.dtype == np.uint8
    assert np.abs(back.astype(int) - pixels.astype(int)).max() <= 1


def test_to_bgr_preserves_shape_for_one_colour():
    swatch = to_bgr(np.array([65.0, 50.0, 200.0]), "lch")
    assert swatch.shape == (3,) and swatch.dtype == np.uint8


def test_a_colour_outside_the_gamut_is_clipped_not_wrapped():
    bgr = to_bgr(np.array([90.0, 120.0, 40.0]), "lch")       # far more chroma than sRGB can show
    assert bgr.dtype == np.uint8 and bgr.max() <= 255


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_only_hue_channels_are_circular():
    circular = {name: space.circular for name, space in SPACES.items() if space.circular}
    assert circular == {"hsv": {"h"}, "hls": {"h"}, "lch": {"h"}}


def test_every_space_has_a_range_per_channel():
    for space in SPACES.values():
        assert len(space.channels) == len(space.ranges) == 3
        assert space.circular <= set(space.channels)


def test_a_channels_position_is_looked_up_by_name():
    assert get_space("hls").index("s") == 2
    with pytest.raises(ValueError, match="no channel"):
        get_space("hls").index("v")


def test_hue_ranges_span_the_circle():
    for name in ("hsv", "hls", "lch"):
        space = get_space(name)
        assert space.ranges[space.index("h")] == (0, 360)


def test_conversion_lands_inside_the_nominal_ranges():
    pixels = sweep()
    for name, space in SPACES.items():
        values = convert(pixels, name)
        for channel, (low, high) in enumerate(space.ranges):
            assert values[:, channel].min() >= low - 1.0, (name, channel)
            assert values[:, channel].max() <= high + 1.0, (name, channel)


def test_normalize_puts_every_channel_on_zero_to_one():
    normalized = normalize(convert(sweep(), "hsv"), "hsv")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    assert normalized[:, 0].max() > 0.9 and normalized[:, 1].max() > 0.9


# ---------------------------------------------------------------------------
# in_arc
# ---------------------------------------------------------------------------


def test_an_ordinary_arc_is_half_open():
    assert in_arc(np.array([21.9, 22.0, 41.9, 42.0]), 22, 42).tolist() == [False, True, True, False]


def test_an_arc_that_wraps_takes_both_sides_of_zero():
    angles = np.array([339.9, 340.0, 359.9, 0.0, 21.9, 22.0])
    assert in_arc(angles, 340, 22).tolist() == [False, True, True, True, True, False]


def test_the_whole_circle_and_the_empty_arc():
    angles = np.linspace(0, 360, 50, endpoint=False)
    assert in_arc(angles, 0, 360).all()
    assert not in_arc(angles, 90, 90).any()
