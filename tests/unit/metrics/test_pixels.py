"""
The shared rule behind every colour metric: read the pixels inside the mask and nothing else.

That is not an optimization -- a mean colour taken over the whole frame is mostly a measurement of the substrate the
specimen was photographed on, and it would vary with the background across a collection while looking exactly like a
biological signal.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics.pixels import masked_pixels
from critterframe.recipes import Segment
from helpers.synthetic import flat, half_and_half


# ---------------------------------------------------------------------------
# Only the organism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction(),
                                    cf.threshold_fractions(cf.hue_thresholds(), unmatched=True)])
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
# Shared contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction(),
                                    cf.threshold_fractions(cf.hue_thresholds(), unmatched=True)])
def test_every_colour_metric_needs_a_mask(metric):
    with pytest.raises(ValueError, match="has no mask yet"):
        metric(Segment(np.zeros((10, 10, 3), np.uint8)))


@pytest.mark.parametrize("metric", [cf.mean_lightness(), cf.mean_color(),
                                    cf.black_fraction(), cf.red_fraction(),
                                    cf.threshold_fractions(cf.hue_thresholds(), unmatched=True)])
def test_every_colour_metric_reports_a_fraction(metric):
    assert metric.unit == "fraction"


# ---------------------------------------------------------------------------
# masked_pixels
# ---------------------------------------------------------------------------


def test_masked_pixels_returns_only_the_masked_pixels():
    segment = half_and_half(top=(10, 20, 30), bottom=(200, 200, 200), shape=(20, 20))
    pixels = masked_pixels(segment)
    assert pixels.shape == (200, 3)
    assert (pixels == (10, 20, 30)).all()


def test_a_cap_samples_the_same_pixels_for_the_same_seed():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    segment = Segment(image, mask=np.ones((40, 40), bool), occurrence_id="test")

    first = masked_pixels(segment, cap=100, seed=3)
    assert len(first) == 100
    assert (first == masked_pixels(segment, cap=100, seed=3)).all()
    assert not (first == masked_pixels(segment, cap=100, seed=4)).all()


def test_a_cap_larger_than_the_mask_takes_every_pixel():
    assert len(masked_pixels(flat((5, 5, 5), shape=(10, 10)), cap=10_000)) == 100


def test_required_false_returns_none_where_there_is_nothing_to_read():
    no_mask = Segment(np.zeros((10, 10, 3), np.uint8))
    empty = Segment(np.zeros((10, 10, 3), np.uint8), mask=np.zeros((10, 10), bool))
    assert masked_pixels(no_mask, required=False) is None
    assert masked_pixels(empty, required=False) is None


def test_required_true_raises_on_an_empty_mask():
    empty = Segment(np.zeros((10, 10, 3), np.uint8), mask=np.zeros((10, 10), bool))
    with pytest.raises(ValueError, match="empty mask"):
        masked_pixels(empty)
