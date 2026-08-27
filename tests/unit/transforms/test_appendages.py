"""
Removing appendages without shortening the body.

Opening alone would strip the legs and also erode tapers and round corners,
systematically shortening every body it touched. The implementation instead uses
the opened result as a SELECTOR -- it decides which pixels are body, and the
final intersection with the input restores their exact original edges.

That distinction is the thing to test: after cleaning, every surviving pixel
must be a pixel that was there before, and the body's extent must not have
shrunk. A test that only checked "the legs are gone" would pass just as happily
on the version that quietly shaves 3% off every measurement.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment

BODY_CENTRE = (150, 150)
BODY_AXES = (25, 70)


def body_with_legs(legs=True, shape=(300, 300)):
    """An ellipse body with thin lines for legs -- the shape the operation is for."""
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, BODY_CENTRE, BODY_AXES, 0, 0, 360, 1, -1)
    if legs:
        for dx in (-80, 80):
            cv2.line(mask, BODY_CENTRE, (BODY_CENTRE[0] + dx, 260), 1, 2)
    return mask.astype(bool)


def a_segment(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


def test_the_legs_go_and_the_body_stays():
    cleaned, info = cf.remove_appendages()(a_segment(body_with_legs()))
    body_only = body_with_legs(legs=False)

    assert 0 < info["removed_fraction"] < 0.4
    # Everything the body occupied is still there.
    assert (cleaned.mask & body_only).sum() / body_only.sum() > 0.98


def test_no_pixel_is_invented():
    """
    The intersection step: a cleaned mask is a SUBSET of what it started from,
    so the boundary is the segmenter's own rather than a dilated approximation
    of it.
    """
    original = body_with_legs()
    cleaned, _info = cf.remove_appendages()(a_segment(original))
    assert not (cleaned.mask & ~original).any()


def test_the_body_is_not_shortened():
    """
    What separates this from a plain opening. The body's extent along its long
    axis must survive to the pixel, or every length measured afterward is
    quietly short.
    """
    original = body_with_legs(legs=False)
    cleaned, _info = cf.remove_appendages()(a_segment(original))

    before = np.nonzero(original)[0]
    after = np.nonzero(cleaned.mask)[0]
    assert after.min() == before.min()
    assert after.max() == before.max()


def test_a_mask_with_no_appendages_passes_through_essentially_unchanged():
    """
    Which matters when the segmenter includes legs inconsistently: otherwise
    cleaned and uncleaned masks would carry different distortions and stop
    being comparable.
    """
    original = body_with_legs(legs=False)
    cleaned, info = cf.remove_appendages()(a_segment(original))
    assert info["removed_fraction"] < 0.02


def test_severed_fragments_are_discarded_not_kept():
    """
    Step two: erosion severs the legs, and only the largest component is the
    body. A second blob elsewhere in the frame is not a second organism -- one
    occurrence is one organism.
    """
    mask = body_with_legs(legs=False).astype(np.uint8)
    cv2.circle(mask, (40, 40), 12, 1, -1)
    cleaned, info = cf.remove_appendages()(a_segment(mask.astype(bool)))

    assert info["n_components"] > 1
    assert not cleaned.mask[20:60, 20:60].any()


def test_the_report_says_how_much_went(caplog):
    _cleaned, info = cf.remove_appendages()(a_segment(body_with_legs()))
    assert info["area_after"] < info["area_before"]
    assert info["removed_fraction"] == pytest.approx(
        1 - info["area_after"] / info["area_before"])
    assert info["degenerate"] is False


def test_a_thin_mask_is_returned_unchanged_and_flagged(caplog):
    """
    A caller must be able to tell "nothing needed removing" from "the whole
    organism was thinner than the kernel and I gave up" -- the flag is the only
    thing that can, since both return the mask they were given.
    """
    thin = np.zeros((200, 200), np.uint8)
    cv2.line(thin, (20, 100), (180, 100), 1, 1)

    with caplog.at_level("WARNING"):
        cleaned, info = cf.remove_appendages()(a_segment(thin.astype(bool)))

    assert info["degenerate"] is True
    assert np.array_equal(cleaned.mask, thin.astype(bool))
    assert "erosion removed everything" in caplog.text


def test_a_bigger_radius_removes_more():
    """
    The radius scales with the mask's area, so one setting works across
    specimen sizes; the parameter tunes how aggressive that is.
    """
    gentle, gentle_info = cf.remove_appendages(relative_radius=0.02)(
        a_segment(body_with_legs()))
    firm, firm_info = cf.remove_appendages(relative_radius=0.12)(
        a_segment(body_with_legs()))

    assert firm_info["radius"] > gentle_info["radius"]
    assert firm.mask.sum() <= gentle.mask.sum()


def test_removing_appendages_moves_no_pixels():
    """
    Erode, keep-largest, and dilate all operate on the same grid, so the
    mapping back to original coordinates is untouched and the image passes
    through unchanged.
    """
    segment = a_segment(body_with_legs())
    cleaned, _info = cf.remove_appendages()(segment)

    assert np.allclose(cleaned.matrix, segment.matrix)
    assert np.array_equal(cleaned.image, segment.image)


def test_an_empty_mask_raises():
    """Nothing to clean is a broken segmentation, not a clean organism."""
    with pytest.raises(ValueError, match="empty mask"):
        cf.remove_appendages()(a_segment(np.zeros((100, 100), bool)))
