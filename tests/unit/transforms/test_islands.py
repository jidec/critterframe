"""
Removing islands: fragments go, the organism is untouched, and nothing moves.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.core.recipes import Segment


def body_with_islands(shape=(300, 300)):
    """An ellipse body, a mid-sized blob (area 900), and a speck (area 16)."""
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (150, 150), (25, 70), 0, 0, 360, 1, -1)
    mask[20:50, 20:50] = 1
    mask[270:274, 270:274] = 1
    return mask.astype(bool)


def body_only(shape=(300, 300)):
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (150, 150), (25, 70), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def a_segment(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


def test_default_keeps_only_the_largest_component():
    cleaned, info = cf.remove_islands()(a_segment(body_with_islands()))

    assert np.array_equal(cleaned.mask, body_only())
    assert info["n_components"] == 3
    assert info["n_removed"] == 2
    assert info["area_before"] - info["area_after"] == 900 + 16


def test_min_area_frac_keeps_large_secondary_components():
    cleaned, info = cf.remove_islands(min_area_frac=0.1)(a_segment(body_with_islands()))

    assert info["n_removed"] == 1
    assert cleaned.mask[35, 35]
    assert not cleaned.mask[272, 272]
    assert cleaned.mask[body_only()].all()


def test_a_single_component_passes_through_unchanged():
    original = body_only()
    cleaned, info = cf.remove_islands()(a_segment(original))

    assert np.array_equal(cleaned.mask, original)
    assert info["n_removed"] == 0
    assert info["removed_fraction"] == 0.0


def test_no_pixel_is_invented_and_nothing_moves():
    original = body_with_islands()
    segment = a_segment(original)
    cleaned, _info = cf.remove_islands(min_area_frac=0.5)(segment)

    assert not (cleaned.mask & ~original).any()
    assert np.array_equal(cleaned.matrix, segment.matrix)


def test_the_threshold_is_in_the_hash():
    assert cf.remove_islands().spec() != cf.remove_islands(min_area_frac=0.1).spec()


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_out_of_range_threshold_is_refused(bad):
    with pytest.raises(ValueError):
        cf.remove_islands(min_area_frac=bad)


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        cf.remove_islands()(a_segment(np.zeros((50, 50), bool)))
