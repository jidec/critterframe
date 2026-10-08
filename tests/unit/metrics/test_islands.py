"""
Counting islands: the fragments of a mask that aren't the organism.

The same thing `remove_islands` removes, counted instead: a mask in one piece
has none, and the organism itself is never one of them.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment


def body(shape=(300, 300)):
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (150, 150), (25, 70), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def a_segment(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


def test_a_mask_in_one_piece_has_no_islands():
    assert cf.n_islands()(a_segment(body())) == 0


def test_each_fragment_besides_the_organism_is_one_island():
    mask = body()
    mask[20:50, 20:50] = True  # a blob
    mask[270:274, 270:274] = True  # a speck
    assert cf.n_islands()(a_segment(mask)) == 2


def test_size_does_not_matter_only_separation():
    """One stray pixel is an island; a large lobe still attached is not."""
    mask = body()
    mask[150:160, 150:260] = True  # attached to the body
    assert cf.n_islands()(a_segment(mask)) == 0

    mask[5, 5] = True
    assert cf.n_islands()(a_segment(mask)) == 1


def test_touching_at_a_corner_is_attached():
    """8-connected, as remove_islands is: a diagonal neighbour is the same piece."""
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True
    mask[5, 5] = True  # diagonal to the block's corner
    assert cf.n_islands()(a_segment(mask)) == 0


def test_it_counts_what_remove_islands_would_remove():
    mask = body()
    mask[20:50, 20:50] = True
    mask[270:274, 270:274] = True
    segment = a_segment(mask)

    _cleaned, info = cf.remove_islands()(segment)
    assert cf.n_islands()(segment) == info["n_removed"]


def test_after_remove_islands_there_are_none():
    mask = body()
    mask[20:50, 20:50] = True
    cleaned, _info = cf.remove_islands()(a_segment(mask))
    assert cf.n_islands()(cleaned) == 0


def test_an_empty_mask_has_no_islands():
    assert cf.n_islands()(a_segment(np.zeros((20, 20), bool))) == 0


def test_a_segment_without_a_mask_is_refused():
    with pytest.raises(ValueError, match="mask"):
        cf.n_islands()(Segment(np.zeros((10, 10, 3), np.uint8)))


def test_it_is_a_count_and_can_be_named():
    assert cf.n_islands().unit == "count"
    assert cf.n_islands().metric_name == "n_islands"
    assert cf.n_islands(name="abdomen_islands").metric_name == "abdomen_islands"
