"""
Quality scores: numbers about the IMAGE rather than about the organism.

These are metrics like any other -- stored the same way, exported the same way,
filtered the same way -- and that is the design point. A blur score is not a
special kind of thing that lives in a quality system; it is a derived value, so
a project can threshold on it at export and revise the threshold tomorrow
without recomputing anything.

Two of them report in ORIGINAL coordinates on purpose (`edge_fraction`,
`mask_fraction`): "is the specimen cut off by the frame" is a question about the
photograph, and a cropped working frame has different edges from the photograph
it came from.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics.quality import WARN_THRESHOLDS
from critterframe.recipes import Segment


def blob(shape=(200, 200), axes=(30, 60), angle=0, centre=None):
    mask = np.zeros(shape, np.uint8)
    centre = centre or (shape[1] // 2, shape[0] // 2)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 1, -1)
    return mask.astype(bool)


def a_segment(mask=None, image=None):
    mask = blob() if mask is None else mask
    if image is None:
        image = np.zeros((*mask.shape, 3), np.uint8)
        image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


def textured(shape=(200, 200)):
    """A high-frequency checkerboard: sharp by construction."""
    grid = np.indices(shape).sum(axis=0) % 2
    return np.repeat((grid * 255).astype(np.uint8)[..., None], 3, axis=2)


# ---------------------------------------------------------------------------
# blur_variance
# ---------------------------------------------------------------------------


def test_a_blurred_image_scores_lower_than_a_sharp_one():
    """
    The only claim worth making about a Laplacian variance. Its absolute value
    depends on the subject as much as on the focus, which is precisely why the
    threshold is calibrated per project (validation.filters) rather than
    hardcoded here.
    """
    sharp = textured()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    mask = np.ones(sharp.shape[:2], bool)

    assert (cf.blur_variance()(a_segment(mask, blurred))
            < cf.blur_variance()(a_segment(mask, sharp)))


def test_blur_is_measured_on_the_organism_not_the_background():
    """
    A sharp leaf behind an out-of-focus moth must not read as a sharp moth.
    """
    image = textured()
    image[blob()] = 128                       # flat, featureless organism
    on_organism = cf.blur_variance()(a_segment(blob(), image))
    on_everything = cf.blur_variance()(a_segment(np.ones(image.shape[:2], bool),
                                                 image))
    assert on_organism < on_everything


# ---------------------------------------------------------------------------
# bilateral_asymmetry
# ---------------------------------------------------------------------------


def test_a_symmetric_mask_scores_near_zero():
    assert cf.bilateral_asymmetry()(a_segment(blob())) < 0.05


def test_an_asymmetric_mask_scores_higher():
    """
    What it is for: a mask with a chunk missing on one side is usually a
    segmentation failure rather than an unusual specimen.
    """
    lopsided = blob().copy()
    lopsided[:, 100:] = False                 # keep only the left half

    # Compared against the symmetric case rather than against an absolute
    # threshold: what the score means is "less like its own mirror image than
    # that one is", and the number a given shape lands on is a property of the
    # shape. Calibrating a cutoff for a project is validation.filters' job.
    assert (cf.bilateral_asymmetry()(a_segment(lopsided))
            > cf.bilateral_asymmetry()(a_segment(blob())) + 0.2)


def test_asymmetry_is_measured_about_the_mask_s_own_centre():
    """
    Not the frame's. A specimen sitting off-centre is not asymmetric, and a
    score that said so would flag every off-centre crop in the project.
    """
    centred = cf.bilateral_asymmetry()(a_segment(blob()))
    offset = cf.bilateral_asymmetry()(a_segment(blob(centre=(40, 100))))
    assert abs(centred - offset) < 0.05


# ---------------------------------------------------------------------------
# edge_fraction
# ---------------------------------------------------------------------------


def test_a_specimen_clear_of_the_border_touches_no_edge():
    assert cf.edge_fraction()(a_segment(blob())) == 0.0


def test_a_specimen_running_off_the_frame_reports_the_touching_fraction():
    """
    "Cut off" is the failure this catches, and it cannot be seen from any size
    trait: a truncated organism measures as a smaller, perfectly plausible one.
    """
    mask = np.zeros((200, 200), bool)
    mask[80:120, 150:] = True                 # runs off the right edge
    assert cf.edge_fraction()(a_segment(mask)) > 0.0


def test_edge_fraction_is_judged_in_original_coordinates():
    """
    A cropped working frame has different edges from the photograph. The
    question is about the photograph, so the mask goes back there first --
    otherwise crop_to_mask() would make every specimen look cut off.
    """
    segment = a_segment(blob())
    cropped, _info = cf.crop_to_mask(pad=0)(segment)

    assert cf.edge_fraction()(cropped) == 0.0


# ---------------------------------------------------------------------------
# mask_fraction
# ---------------------------------------------------------------------------


def test_mask_fraction_is_the_share_of_the_original_frame():
    mask = np.zeros((100, 100), bool)
    mask[:50, :] = True
    assert cf.mask_fraction()(a_segment(mask)) == pytest.approx(0.5)


def test_mask_fraction_ignores_a_crop_for_the_same_reason():
    segment = a_segment(blob())
    cropped, _info = cf.crop_to_mask(pad=0)(segment)
    assert cf.mask_fraction()(cropped) == pytest.approx(
        cf.mask_fraction()(segment))


# ---------------------------------------------------------------------------
# Shared contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [cf.blur_variance(), cf.bilateral_asymmetry(),
                                    cf.edge_fraction(), cf.mask_fraction()])
def test_every_quality_metric_refuses_an_empty_mask(metric):
    with pytest.raises(ValueError, match="empty mask"):
        metric(a_segment(np.zeros((50, 50), bool)))


@pytest.mark.parametrize("metric, unit", [
    (cf.blur_variance(), "laplacian_var"),
    (cf.bilateral_asymmetry(), "fraction"),
    (cf.edge_fraction(), "fraction"),
    (cf.mask_fraction(), "fraction"),
])
def test_units_say_what_the_number_is(metric, unit):
    """
    A bare number whose unit lives only in a variable name is the easiest thing
    in this package to misread later -- and "laplacian_var" is not convertible
    to millimetres, which is the export's business to know.
    """
    assert metric.unit == unit


def test_warn_thresholds_are_stated_rather_than_applied():
    """
    They exist to annotate a panel, not to filter. A judgement about degree
    belongs at export, where it stays revisable.
    """
    assert set(WARN_THRESHOLDS) >= {"blur_variance", "bilateral_asymmetry",
                                    "edge_fraction"}


class RecordingSink:
    def __init__(self):
        self.stages = []

    def collect(self, occurrence_id, stage, image):
        self.stages.append(stage)
        assert image.dtype == np.uint8


@pytest.mark.parametrize("metric, stage", [
    (cf.blur_variance(), "blur_variance"),
    (cf.bilateral_asymmetry(), "bilateral_asymmetry"),
    (cf.edge_fraction(), "edge_fraction"),
])
def test_each_score_draws_the_evidence_for_itself(metric, stage):
    sink = RecordingSink()
    mask = blob()
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = 200
    metric(Segment(image, mask=mask, occurrence_id="test", panel_sink=sink))
    assert sink.stages == [stage]
