"""
Colour thresholds: named cutoffs on colour channels, and the fraction of an organism past them.

A threshold is identified by its spec alone -- two thresholds with the same cutoffs are the same work, and any change
to a cutoff is a different recipe. "How much of this organism is black" is comparable within a project shot on one rig, and comparable across rigs only
once colour calibration exists.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.colorspaces import to_bgr
from critterframe.metrics import color_thresholds
from critterframe.metrics.color_thresholds import HUE_BANDS, ColorThreshold, color_threshold, threshold_masks
from critterframe.recipes import Segment
from helpers.synthetic import flat


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


def hue_swatch(degrees):
    """A fully saturated, fully bright organism of one hue."""
    return flat(tuple(int(c) for c in to_bgr(np.array([degrees, 1.0, 1.0]), "hsv")))


@pytest.mark.parametrize("hue", sorted(HUE_BANDS))
def test_every_band_recognises_its_own_centre(hue):
    start, end = HUE_BANDS[hue]
    centre = (start + ((end - start) % 360) / 2) % 360
    assert cf.hue_fraction(hue)(hue_swatch(centre)) > 0.9
    for other in HUE_BANDS.keys() - {hue}:
        assert cf.hue_fraction(other)(hue_swatch(centre)) < 0.1


@pytest.mark.parametrize("degrees", [350, 355, 359, 0, 5, 15])
def test_red_is_one_band_across_zero_degrees(degrees):
    assert cf.red_fraction()(hue_swatch(degrees)) > 0.9


# ---------------------------------------------------------------------------
# ColorThreshold: construction
# ---------------------------------------------------------------------------


def test_keywords_and_a_nested_mapping_build_the_same_threshold():
    by_keyword = color_threshold("yellow", hsv_h=(42, 72), hsv_s=(0.25, None), lab_l=(20, None))
    by_mapping = ColorThreshold("yellow", {"lab": {"l": (20, None)}, "hsv": {"s": (0.25, None), "h": (42, 72)}})
    assert by_keyword == by_mapping
    assert by_keyword.spaces == ("hsv", "lab")


def test_a_spec_round_trips_and_is_plain_json():
    threshold = color_threshold("warm", lch_h=(340, 60), lch_c=(15, None))
    spec = threshold.spec()
    assert spec == {"name": "warm", "conditions": {"lch": {"c": [15.0, None], "h": [340.0, 60.0]}}}
    assert ColorThreshold.from_spec(spec) == threshold


def test_changing_any_cutoff_changes_the_spec():
    assert (color_threshold("x", lab_l=(None, 20)).spec()
            != color_threshold("x", lab_l=(None, 21)).spec())


@pytest.mark.parametrize("build, message", [
    (lambda: color_threshold("x", cmyk_c=(0, 1)), "unknown colour space"),
    (lambda: color_threshold("x", hsv_q=(0, 1)), "no channel"),
    (lambda: color_threshold("x", hsvh=(0, 1)), "<space>_<channel>"),
    (lambda: color_threshold("x", lab_l=(50, 20)), "not below"),
    (lambda: color_threshold("x", lab_l=(50, 50)), "not below"),
    (lambda: color_threshold("x", lab_l=(None, None)), "at least one bound"),
    (lambda: color_threshold("x", hsv_h=(340, None)), "both ends"),
    (lambda: color_threshold("x", hsv_h=(90, 90)), "empty arc"),
    (lambda: color_threshold("x", hsv_h=(-10, 20)), "0-360"),
    (lambda: color_threshold("x", lab_l=(1, 2, 3)), r"\(low, high\)"),
    (lambda: color_threshold("x"), "no conditions"),
    (lambda: color_threshold("", lab_l=(None, 20)), "non-empty"),
    (lambda: color_threshold("dark__patch", lab_l=(None, 20)), "'__'"),
])
def test_a_malformed_threshold_fails_when_it_is_built(build, message):
    with pytest.raises(ValueError, match=message):
        build()


# ---------------------------------------------------------------------------
# ColorThreshold: what it matches
# ---------------------------------------------------------------------------


def test_bounds_are_half_open_and_none_is_open():
    values = np.array([[10.0], [20.0], [30.0]])
    converted = {"lab": np.hstack([values, np.zeros((3, 2))])}
    assert color_threshold("x", lab_l=(20, 30)).matches(converted).tolist() == [False, True, False]
    assert color_threshold("x", lab_l=(None, 20)).matches(converted).tolist() == [True, False, False]
    assert color_threshold("x", lab_l=(20, None)).matches(converted).tolist() == [False, True, True]


def test_a_hue_arc_wraps_through_zero():
    hues = np.array([[330.0], [350.0], [5.0], [30.0]])
    converted = {"lch": np.hstack([np.full((4, 2), 50.0), hues])}
    assert color_threshold("x", lch_h=(340, 22)).matches(converted).tolist() == [False, True, True, False]


def test_conditions_across_spaces_must_all_hold():
    """Bright and red: lab lightness AND an hsv hue arc, each read in its own space."""
    bright_red, dark_red, bright_green = (80, 80, 255), (0, 0, 90), (80, 255, 80)
    pixels = np.array([bright_red, dark_red, bright_green], np.uint8)
    threshold = color_threshold("bright_red", lab_l=(50, None), hsv_h=(340, 22))
    assert threshold_masks(pixels, [threshold])["bright_red"].tolist() == [True, False, False]


def test_each_space_is_converted_once_however_many_thresholds_read_it(monkeypatch):
    calls = []
    real = color_thresholds.convert

    def counting(pixels, space):
        calls.append(space)
        return real(pixels, space)

    monkeypatch.setattr(color_thresholds, "convert", counting)
    threshold_masks(np.zeros((5, 3), np.uint8), cf.hue_thresholds() + [color_threshold("dark", lab_l=(None, 20))])
    assert sorted(calls) == ["hsv", "lab"]


def test_masks_work_on_a_whole_image_too():
    image = np.zeros((4, 6, 3), np.uint8)
    image[:2] = 255
    masks = threshold_masks(image, [color_threshold("light", lab_l=(50, None))])
    assert masks["light"].shape == (4, 6)
    assert masks["light"][:2].all() and not masks["light"][2:].any()


# ---------------------------------------------------------------------------
# threshold_fractions
# ---------------------------------------------------------------------------


def test_fractions_are_reported_per_threshold_in_the_order_given():
    image = np.zeros((2, 2, 3), np.uint8)
    image[1, 1] = 255                                     # 3 dark pixels, 1 light
    segment = Segment(image, mask=np.ones((2, 2), bool))

    fractions = cf.threshold_fractions([color_threshold("light", lab_l=(50, None)),
                                        color_threshold("dark", lab_l=(None, 50))])(segment)
    assert list(fractions) == ["light", "dark"]
    assert fractions == pytest.approx({"light": 0.25, "dark": 0.75})


def test_overlapping_thresholds_each_count_every_pixel_they_match():
    """Independent, not a partition: a pixel two thresholds share counts toward both."""
    fractions = cf.threshold_fractions([color_threshold("not_black", lab_l=(20, None)),
                                        color_threshold("not_white", lab_l=(None, 90))])(flat((128, 128, 128)))
    assert fractions == {"not_black": 1.0, "not_white": 1.0}


def test_unmatched_is_the_share_no_threshold_matched():
    """Half red, half grey: grey matches no hue, so unmatched is exactly the grey half."""
    image = np.zeros((10, 10, 3), np.uint8)
    image[:5] = (0, 0, 255)
    image[5:] = (128, 128, 128)
    segment = Segment(image, mask=np.ones((10, 10), bool))

    fractions = cf.threshold_fractions(cf.hue_thresholds(), unmatched=True)(segment)
    assert fractions["red"] == pytest.approx(0.5)
    assert fractions["unmatched"] == pytest.approx(0.5)
    assert list(fractions)[-1] == "unmatched"


def test_unmatched_is_off_by_default():
    assert "unmatched" not in cf.threshold_fractions(cf.hue_thresholds())(flat((0, 0, 255)))


def test_the_hue_palette_is_the_six_hue_bands():
    assert [threshold.name for threshold in cf.hue_thresholds()] == list(HUE_BANDS)


@pytest.mark.parametrize("thresholds, unmatched, message", [
    ([], False, "at least one"),
    ([color_threshold("a", lab_l=(None, 20)), color_threshold("a", lab_l=(80, None))], False, "unique"),
    ([color_threshold("unmatched", lab_l=(None, 20))], True, "collides"),
])
def test_an_ambiguous_set_of_thresholds_raises(thresholds, unmatched, message):
    with pytest.raises(ValueError, match=message):
        cf.threshold_fractions(thresholds, unmatched=unmatched)


def test_the_thresholds_and_the_unmatched_flag_are_hashed():
    dark = color_threshold("dark", lab_l=(None, 20))
    assert (cf.threshold_fractions([dark]).spec()
            != cf.threshold_fractions([dark], unmatched=True).spec())
    assert (cf.threshold_fractions([dark]).spec()
            != cf.threshold_fractions([color_threshold("dark", lab_l=(None, 25))]).spec())


def test_a_visualized_measurement_draws_one_picture_per_threshold():
    """Side by side, one per threshold plus unmatched; matched pixels keep their own colour, the rest go grey."""
    image = np.zeros((80, 60, 3), np.uint8)       # tall enough that the caption stays clear of the rows sampled
    image[:40] = (0, 0, 255)
    image[40:] = (200, 200, 200)
    segment = Segment(image, mask=np.ones((80, 60), bool))
    masks, fractions = color_thresholds._measure(
        segment, [color_threshold("red", hsv_h=(340, 22), hsv_s=(0.25, None))], unmatched=True)

    panel = color_thresholds.threshold_panel(segment, masks, fractions)
    assert panel.shape == (80, 120, 3)
    red_picture, unmatched_picture = panel[:, :60], panel[:, 60:]
    assert (red_picture[30:40] == (0, 0, 255)).all()                                  # matched: own colour
    assert (red_picture[40:] == red_picture[79, 0]).all() and red_picture[79, 0].max() < 100   # rest: dim grey
    assert (unmatched_picture[40:] == (200, 200, 200)).all()
