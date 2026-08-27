"""
Orientation: finding the body axis, and knowing when it can't be found.

The axis is chosen by ASYMMETRY rather than by length, and that is the whole
idea. Length is the obvious criterion and the wrong one: a spread-wing moth's
longest axis is its wingspan, not its body. An insect's head end and abdomen end
differ; its left and right wingtips are near mirror images. So both principal
components are scored by the skewness of the pixel cloud projected onto them,
and the asymmetric one wins.

The tests use drawn shapes with a known answer, including the two cases that
matter most: a shape whose long axis is NOT its body, and a shape with no
meaningful axis at all -- where the honest output is a flag saying so, not a
confident number.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.transforms.orient import (
    apply_affine,
    compute_orientation,
    rotation_matrix,
)


def tapered_body(angle_deg=0, shape=(300, 300)):
    """
    A body that is asymmetric along its length -- wide at one end, narrow at the
    other. That asymmetry is the signal orientation is built to find.
    """
    mask = np.zeros(shape, np.uint8)
    centre = (shape[1] // 2, shape[0] // 2)
    for offset in range(-70, 71):
        half_width = int(22 - 0.22 * (offset + 70) / 2)
        cv2.line(mask, (centre[0] - half_width, centre[1] + offset),
                 (centre[0] + half_width, centre[1] + offset), 1, 1)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
        mask = cv2.warpAffine(mask, matrix, (shape[1], shape[0]),
                              flags=cv2.INTER_NEAREST)
    return mask.astype(bool)


def winged_body(shape=(300, 300)):
    """
    A tapered body with symmetric wings twice as wide as it is long: the case
    where the longest axis is emphatically not the body.
    """
    mask = tapered_body(shape=shape).astype(np.uint8)
    centre = (shape[1] // 2, shape[0] // 2)
    cv2.ellipse(mask, centre, (140, 16), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def a_segment(mask):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


# ---------------------------------------------------------------------------
# compute_orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0, 20, 45, -30, 75])
def test_a_tilted_body_is_rotated_back_upright(angle):
    """
    The rotation reported must undo the tilt, to within the discretization of a
    drawn shape. Modulo 180: which END is up is a separate question from which
    axis is the body.
    """
    rotation_deg, _cx, _cy, _info = compute_orientation(tapered_body(angle))
    assert min(abs((rotation_deg - angle) % 180),
               180 - abs((rotation_deg - angle) % 180)) < 6


def test_the_body_wins_over_the_longer_wingspan():
    """
    THE property. The wings are the longest axis and the body is the asymmetric
    one, so a length-based choice would report the wingspan and every body
    length measured afterward would be a wingspan.
    """
    _rotation, _cx, _cy, info = compute_orientation(winged_body())
    assert info["chose_longer_axis"] is False


def test_choosing_by_length_instead_is_available_and_reports_the_wingspan():
    """
    The flag exists so a project whose organisms really are symmetric can say
    so -- and this is what it does.
    """
    _rotation, _cx, _cy, info = compute_orientation(
        winged_body(), body_axis_is_higher_skew=False)
    assert info["chose_longer_axis"] is True


def test_the_centroid_is_the_rotation_centre():
    mask = tapered_body()
    _rotation, cx, cy, _info = compute_orientation(mask)
    ys, xs = np.nonzero(mask)
    assert abs(cx - xs.mean()) < 1e-6 and abs(cy - ys.mean()) < 1e-6


def test_a_round_mask_is_flagged_unreliable():
    """
    A circle has no axis. The eigenvalue ratio near 1 says the answer is noise,
    and reporting that is more useful than a confident number -- which is
    exactly what a symmetric synthetic specimen produces.
    """
    circle = np.zeros((200, 200), np.uint8)
    cv2.circle(circle, (100, 100), 60, 1, -1)
    _rotation, _cx, _cy, info = compute_orientation(circle.astype(bool))

    assert info["unreliable"] is True
    assert info["eigval_ratio"] > 0.9


def test_an_elongated_mask_is_not_flagged():
    _rotation, _cx, _cy, info = compute_orientation(tapered_body())
    assert info["unreliable"] is False


def test_the_skews_of_both_axes_are_reported():
    """Diagnostics a caller can check rather than a decision it must trust."""
    _rotation, _cx, _cy, info = compute_orientation(tapered_body())
    assert {"skew_pc0", "skew_pc1", "chosen_pc"} <= set(info)
    assert info["chosen_pc"] in (0, 1)


def test_an_empty_mask_raises():
    with pytest.raises(ValueError, match="empty mask"):
        compute_orientation(np.zeros((10, 10), bool))


# ---------------------------------------------------------------------------
# rotation_matrix / apply_affine
# ---------------------------------------------------------------------------


def test_the_canvas_expands_to_fit_the_rotated_corners():
    """Nothing is clipped, which for a mask would mean losing organism."""
    _matrix, size = rotation_matrix(45, 50, 50, (100, 100))
    assert size[0] > 100 and size[1] > 100


def test_no_rotation_needs_no_expansion():
    _matrix, size = rotation_matrix(0, 50, 50, (100, 100))
    assert size == (100, 100)


def test_a_mask_survives_the_affine_as_integers():
    """
    Nearest-neighbour by default: interpolating a mask would invent boundary
    pixels that are neither inside nor out.
    """
    mask = tapered_body()
    matrix, size = rotation_matrix(30, 150, 150, mask.shape)
    warped = apply_affine(mask.astype(np.uint8), matrix, size)
    assert set(np.unique(warped)) <= {0, 1}


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def test_orienting_makes_the_body_vertical():
    """
    A vertical body is taller than it is wide once cropped to itself -- which is
    the point of orienting before measuring, since it makes length and width
    mean the same thing across specimens.
    """
    oriented, _info = cf.orient()(a_segment(tapered_body(50)))
    ys, xs = np.nonzero(oriented.mask)
    assert (ys.max() - ys.min()) > (xs.max() - xs.min())


def test_the_operation_reports_what_it_decided():
    _oriented, info = cf.orient()(a_segment(tapered_body(20)))
    assert "rotation_deg" in info
    assert "unreliable" in info


def test_orienting_moves_pixels_and_says_so():
    """
    Which means it must compose onto the affine -- a mask found after orienting
    has to invert back to the specimen's real position in the original frame.
    """
    segment = a_segment(tapered_body(30))
    oriented, _info = cf.orient()(segment)

    assert not np.allclose(oriented.matrix, segment.matrix)
    restored = oriented.mask_in_original_coordinates()
    assert restored.shape == segment.shape
    overlap = (restored & segment.mask).sum() / (restored | segment.mask).sum()
    assert overlap > 0.9


def test_orienting_needs_a_mask():
    image = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError, match="has no mask yet"):
        cf.orient()(Segment(image, occurrence_id="test"))
