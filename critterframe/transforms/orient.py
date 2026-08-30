"""
PCA orientation, axis chosen by asymmetry rather than length.

Length alone picks the wingspan on a spread specimen; asymmetry picks the body,
since a head-to-tail axis is lopsided and a wingtip-to-wingtip one is not. A
near-isotropic mask has numerically unstable axes and is flagged unreliable
rather than silently rotated.
"""

import logging

import cv2
import numpy as np

from ..recipes import Transform
from ..visualization.panels import annotate, side_by_side

logger = logging.getLogger(__name__)

# Which principal component is the body axis: the more asymmetric one. Along
# the body, the two ends differ (blunt thorax vs tapering abdomen) -> high
# |skew|. Along the wing axis the two ends are near mirror images -> low |skew|.
# Flip this if your masks say otherwise.
BODY_AXIS_IS_HIGHER_SKEW = True

# Below this eigenvalue ratio (minor/major) the mask is too close to isotropic
# for the axes to be meaningful -- orientation should be treated as unreliable.
ISOTROPY_WARN_RATIO = 0.8


def orient(body_axis_is_higher_skew=BODY_AXIS_IS_HIGHER_SKEW,
           isotropy_warn_ratio=ISOTROPY_WARN_RATIO):
    """
    Operation: rotate the segment so the body axis is vertical.

    body_axis_is_higher_skew -- pick the body axis as the more asymmetric
                                principal component (see module constants).
    isotropy_warn_ratio      -- eigenvalue ratio above which the orientation is
                                reported unreliable.
    """
    return Transform("orient", _orient, {
        "body_axis_is_higher_skew": body_axis_is_higher_skew,
        "isotropy_warn_ratio": isotropy_warn_ratio,
    }, version="1")


def _skew(values):
    """Fisher skewness of a 1D array; 0 if degenerate."""
    sd = values.std()
    if sd == 0:
        return 0.0
    return float((((values - values.mean()) / sd) ** 3).mean())


def compute_orientation(mask, body_axis_is_higher_skew=BODY_AXIS_IS_HIGHER_SKEW,
                        isotropy_warn_ratio=ISOTROPY_WARN_RATIO):
    """
    Find the body axis of a boolean mask via PCA, choosing between the two
    principal components by asymmetry rather than by length.

    Length is the obvious criterion and the wrong one: a spread-wing moth's
    longest axis is its wingspan, not its body. Both PCs are instead scored by
    the skewness of the pixel distribution projected onto them, and the body
    axis is the asymmetric one -- an insect's head end and abdomen end differ,
    while its left and right wingtips are near mirror images.

    Returns (rotation_deg, cx, cy, info):
      rotation_deg -- rotation to make the body axis vertical (positive = CCW)
      cx, cy       -- centroid, the rotation center, in (x, y) pixels
      info         -- diagnostics: the skew of each PC, which was chosen,
                      whether the chosen axis was the longer one, and the
                      eigenvalue ratio (near 1 = orientation unreliable)
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")

    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy

    covariance = np.cov(np.vstack([dx, dy]))
    eigvals, eigvecs = np.linalg.eigh(covariance)   # ascending

    # project the pixel cloud onto each PC and score its asymmetry
    skews = []
    for index in (0, 1):
        axis = eigvecs[:, index]
        skews.append(_skew(axis[0] * dx + axis[1] * dy))

    abs_skews = [abs(value) for value in skews]
    chosen = int(np.argmax(abs_skews)) if body_axis_is_higher_skew \
        else int(np.argmin(abs_skews))

    axis = eigvecs[:, chosen]
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    rotation_deg = 90.0 - angle

    # eigenvalue ratio: minor/major. Near 1 means the mask is nearly isotropic
    # and the axis directions are numerically unstable.
    ratio = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else 1.0

    info = {
        "skew_pc0": skews[0],
        "skew_pc1": skews[1],
        "chosen_pc": chosen,
        "chose_longer_axis": chosen == 1,   # index 1 == largest eigenvalue
        "eigval_ratio": ratio,
        "unreliable": ratio > isotropy_warn_ratio,
    }
    return rotation_deg, float(cx), float(cy), info


def rotation_matrix(rotation_deg, cx, cy, shape):
    """
    The affine that rotates an array of `shape` by rotation_deg about (cx, cy),
    EXPANDING the canvas to fit the rotated result rather than keeping the
    original size, plus that new (width, height).

    An image is only as tall/wide as the specimen needed lying in its ORIGINAL
    orientation -- a mostly-horizontal insect rotated upright needs a canvas
    taller than the original height, or the head and tail end up past the old
    frame edge and warpAffine silently clips them. Sizing the canvas to the
    rotated bounding box of the original frame guarantees the whole image
    survives regardless of angle or how off-center (cx, cy) is.
    """
    height, width = shape[:2]
    matrix = cv2.getRotationMatrix2D((cx, cy), -rotation_deg, 1.0)

    # Where the four original corners land after rotation -- size the new canvas
    # to their bounding box, then shift so nothing falls negative.
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]],
                       dtype=np.float64)
    rotated = corners @ matrix[:, :2].T + matrix[:, 2]
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)

    new_width, new_height = np.ceil(max_xy - min_xy).astype(int)
    matrix[:, 2] -= min_xy

    return matrix, (int(new_width), int(new_height))


def apply_affine(array, matrix, size, flags=cv2.INTER_NEAREST):
    """
    Warp an array by a 2x3 affine into a (width, height) canvas, padding with
    zeros -- harmless for both a mask (False) and an image (black).

    flags -- interpolation. INTER_NEAREST for masks (no new values, stays
             binary); INTER_LINEAR for images where smoothing is acceptable.
    """
    return cv2.warpAffine(array, matrix, size, flags=flags)


def _orient(segment, body_axis_is_higher_skew=BODY_AXIS_IS_HIGHER_SKEW,
            isotropy_warn_ratio=ISOTROPY_WARN_RATIO):
    """
    Rotate a segment's mask AND image so the body axis is vertical, by the
    identical rotation and center so the two stay pixel-aligned -- a rotated
    mask over an unrotated image lines up with nothing, and every metric
    reading colour under the mask would be reading the wrong pixels.
    """
    mask = segment.require_mask()
    rotation_deg, cx, cy, info = compute_orientation(
        mask, body_axis_is_higher_skew=body_axis_is_higher_skew,
        isotropy_warn_ratio=isotropy_warn_ratio,
    )

    matrix, size = rotation_matrix(rotation_deg, cx, cy, mask.shape)
    oriented_mask = apply_affine(mask.astype(np.uint8), matrix, size) > 0
    oriented_image = apply_affine(segment.image, matrix, size,
                                  flags=cv2.INTER_LINEAR)

    info["rotation_deg"] = rotation_deg

    oriented = segment.replace(image=oriented_image, mask=oriented_mask,
                               applied=matrix)
    _visualize(segment, oriented, rotation_deg, cx, cy, info)
    return oriented, info


def _visualize(segment, oriented, rotation_deg, cx, cy, info):
    """
    Side by side: the original with the chosen axis (green) and the REJECTED
    axis (blue) drawn, next to the oriented result. Seeing both axes is the
    point -- it shows whether the asymmetry test picked the body over the
    wingspan, which a single drawn axis wouldn't tell you.
    """
    if segment.panel_sink is None:
        return

    before = np.asarray(segment.image).copy()
    after = np.asarray(oriented.image).copy()
    length = max(segment.shape)

    def draw_axis(image, theta_deg, color, thickness):
        theta = np.radians(theta_deg)
        ax, ay = np.cos(theta), np.sin(theta)
        cv2.line(image,
                 (int(cx - ax * length), int(cy - ay * length)),
                 (int(cx + ax * length), int(cy + ay * length)),
                 color, thickness)

    chosen_angle = 90.0 - rotation_deg
    draw_axis(before, chosen_angle + 90.0, (255, 128, 0), 1)  # rejected: blue
    draw_axis(before, chosen_angle, (0, 255, 0), 2)           # chosen: green
    cv2.circle(before, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    annotate(before, f"pc{info['chosen_pc']} "
                     f"skew {info['skew_pc0']:+.2f}/{info['skew_pc1']:+.2f} "
                     f"ratio {info['eigval_ratio']:.2f} rot {rotation_deg:+.1f}")
    if info["unreliable"]:
        annotate(before, "UNRELIABLE", line=1, color=(0, 0, 255))

    segment.emit_panel(side_by_side(before, after), "orient")
