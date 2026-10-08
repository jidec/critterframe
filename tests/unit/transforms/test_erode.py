"""
Eroding a mask: in from every edge, by a share of its own thickness.

The share is of the maximum inscribed radius, because erosion works against
thickness. A share of length, or of the square root of area, takes far more out
of a long thin part than a round one of the same size.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment


def disc(radius=40, shape=(200, 200), centre=(100, 100)):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, centre, radius, 1, -1)
    return mask.astype(bool)


def bar(width=20, length=160, shape=(200, 200)):
    mask = np.zeros(shape, bool)
    top = (shape[0] - length) // 2
    left = (shape[1] - width) // 2
    mask[top:top + length, left:left + width] = True
    return mask


def a_segment(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


def thickness(mask):
    """The widest row of a mask, in pixels."""
    return int(mask.sum(axis=1).max())


def test_the_default_takes_a_tenth_of_the_radius_off_every_edge():
    eroded, info = cf.erode()(a_segment(disc(radius=40)))

    assert info["inscribed_radius_px"] == pytest.approx(40, abs=1.5)
    assert info["radius_px"] == pytest.approx(4, abs=0.2)
    assert thickness(eroded.mask) == pytest.approx(2 * 36, abs=3)


def test_pixels_can_be_given_instead():
    eroded, info = cf.erode(px=10)(a_segment(disc(radius=40)))

    assert info["radius_px"] == 10
    assert thickness(eroded.mask) == pytest.approx(2 * 30, abs=3)


def test_a_thin_part_and_a_round_one_lose_the_same_share_of_their_thickness():
    """
    The reason for the measure. A bar 20 wide and a disc 80 across both lose
    about a tenth of their thickness, where a share of sqrt(area) would take
    three times as much of the bar's width as of the disc's.
    """
    round_before, thin_before = disc(radius=40), bar(width=20, length=160)
    round_after, _info = cf.erode(0.1)(a_segment(round_before))
    thin_after, _info = cf.erode(0.1)(a_segment(thin_before))

    round_share = 1 - thickness(round_after.mask) / thickness(round_before)
    thin_share = 1 - thickness(thin_after.mask) / thickness(thin_before)
    assert round_share == pytest.approx(0.1, abs=0.04)
    assert thin_share == pytest.approx(0.1, abs=0.1)        # one pixel is 5% of a 20px bar
    assert abs(round_share - thin_share) < 0.12


def test_it_only_ever_removes_and_touches_nothing_else():
    """No pixels move: the image and the affine are as they were, and the mask never grows."""
    segment = a_segment(bar())
    eroded, info = cf.erode(0.3)(segment)

    assert not (eroded.mask & ~segment.mask).any()
    assert eroded.image is segment.image
    assert np.array_equal(eroded.matrix, segment.matrix)
    assert info["area_after"] < info["area_before"]
    assert info["removed_fraction"] == pytest.approx(
        1 - info["area_after"] / info["area_before"])


def test_a_fraction_can_never_erase_the_mask():
    """Its thickest point is always further from the edge than a share of that same distance."""
    for fraction in (0.01, 0.5, 0.99):
        eroded, info = cf.erode(fraction)(a_segment(bar(width=6)))
        assert eroded.mask.any() and not info["degenerate"]


def test_a_thin_neck_is_cut_through_and_the_info_says_so():
    """
    Erosion is uniform, so a section narrower than twice the radius goes
    entirely. Two lobes joined by a neck come back as two pieces.
    """
    mask = disc(radius=30, centre=(60, 100)) | disc(radius=30, centre=(140, 100))
    mask[97:103, 60:140] = True                # a neck 6 pixels wide
    eroded, info = cf.erode(px=5)(a_segment(mask))

    assert info["n_components"] == 2
    assert cf.n_islands()(eroded) == 1


def test_pixels_past_the_thickness_leave_the_mask_alone_and_say_so():
    """Returned unchanged and flagged, like every transform that can't do what it was asked."""
    segment = a_segment(bar(width=10))
    eroded, info = cf.erode(px=50)(segment)

    assert info["degenerate"] is True
    assert np.array_equal(eroded.mask, segment.mask)
    assert info["removed_fraction"] == 0.0


def test_the_frame_edge_counts_as_an_edge():
    """A mask running off the frame is eroded on that side too, the same as on every other."""
    mask = np.zeros((100, 100), bool)
    mask[:, 30:70] = True                      # touches the top and bottom of the frame
    eroded, _info = cf.erode(px=5)(a_segment(mask))

    assert not eroded.mask[:5].any() and not eroded.mask[-5:].any()
    assert eroded.mask[50, 50]


def test_an_empty_mask_is_refused():
    with pytest.raises(ValueError, match="empty mask"):
        cf.erode()(a_segment(np.zeros((20, 20), bool)))


@pytest.mark.parametrize("kwargs", [{"fraction": 0}, {"fraction": 1}, {"fraction": 1.5},
                                    {"px": 0}, {"px": -2}])
def test_an_erosion_that_makes_no_sense_fails_before_it_runs(kwargs):
    with pytest.raises(ValueError, match="must be"):
        cf.erode(**kwargs)


def test_how_much_is_eroded_is_part_of_the_recipe():
    assert cf.erode().spec() == cf.erode(0.1).spec()
    assert cf.erode().spec() != cf.erode(0.2).spec()
    assert cf.erode().spec() != cf.erode(px=3).spec()
    assert cf.erode().spec()["parameters"] == {"fraction": 0.1, "px": None}


def test_where_it_sits_in_a_chain_does_not_change_what_it_removes():
    """
    The measure is rotation-invariant, so eroding before or after cropping and
    orienting leaves (to within resampling) the same pixels of the original.
    """
    mask = np.zeros((200, 300), np.uint8)
    cv2.ellipse(mask, (160, 100), (90, 25), 30, 0, 360, 1, -1)
    segment = a_segment(mask.astype(bool))

    first, _info = cf.erode(0.2)(segment)
    moved = segment
    for operation in (cf.crop_to_mask(), cf.orient(), cf.erode(0.2)):
        moved, _info = operation(moved)

    before, after = first.mask, moved.mask_in_original_coordinates()
    overlap = (before & after).sum() / (before | after).sum()
    assert overlap > 0.93


class Sink:
    def __init__(self):
        self.panels = []

    def collect(self, occurrence_id, stage, image):
        self.panels.append((stage, image))


def test_a_panel_shows_the_rim_that_went():
    segment = a_segment(disc())
    cf.erode()(segment)                        # no sink: nothing to emit to, and no error

    sink = Sink()
    segment.panel_sink = sink
    cf.erode(0.3)(segment)

    (stage, panel), = sink.panels
    assert stage == "erode"
    assert panel.dtype == np.uint8 and panel.shape == (*segment.shape, 3)
    assert ((panel == (0, 0, 255)).all(axis=2)).any()     # the eroded rim, in red
