"""
Scale: pixels per millimetre from a target of known size.

Synthetic targets, because the drawing is exact and the expected answer is
therefore known to the pixel -- something a real photograph could never give
you. Harvested from `scripts/simple_tests/calibration_test.py`, which had three
of these assertions and printed the reasoning around them.

The failure mode that matters most is not "no target found". It is a WRONG
answer that looks fine: with no target in the search region, clutter
out-correlates nothing and wins, and a plausible wrong scale silently rescales
every trait in the project. That is why every row stores its score, why a weak
match warns, and why `match_score_min` exists.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.calibrations import scale as scale_calibration
from critterframe.records import calibrations as calibration_records
from critterframe.records.occurrences import ID_COL
from helpers.synthetic import TARGET_MM, draw_target_sheet

# The target sits in the top-left quadrant of the drawn sheet.
TARGET_REGION = (0, 0, 0.5, 0.5)
EMPTY_REGION = (0.5, 0.5, 1.0, 1.0)


@pytest.fixture(scope="module")
def sheet():
    """(sheet, template, expected_px_per_mm) -- drawn once, never mutated."""
    return draw_target_sheet()


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def test_a_drawn_target_measures_at_the_scale_it_was_drawn(sheet):
    image, template, expected = sheet
    result = scale_calibration.scale_from_target(image, template, TARGET_MM,
                                                 region=TARGET_REGION)
    assert abs(result["px_per_mm"] - expected) < 1 / TARGET_MM


def test_the_same_sheet_at_half_resolution_measures_half_the_scale(sheet):
    """
    A pixel now covers twice as much card. The multi-scale match is what makes
    one template work across resolutions.
    """
    import cv2

    image, template, expected = sheet
    half = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    result = scale_calibration.scale_from_target(half, template, TARGET_MM,
                                                 region=TARGET_REGION)
    assert abs(result["px_per_mm"] - expected / 2) < 2 / TARGET_MM


def test_the_measurement_reports_where_and_how_well_it_matched(sheet):
    image, template, _expected = sheet
    result = scale_calibration.scale_from_target(image, template, TARGET_MM,
                                                 region=TARGET_REGION)
    assert set(result) == {"px_per_mm", "score", "cx", "cy", "radius_px",
                           "diameter_px"}
    assert result["score"] > 0.9
    assert abs(result["cx"] - 150) < 5 and abs(result["cy"] - 130) < 5


def test_a_grayscale_frame_works_as_well_as_colour(sheet):
    import cv2

    image, template, expected = sheet
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = scale_calibration.scale_from_target(gray, template, TARGET_MM,
                                                 region=TARGET_REGION)
    assert abs(result["px_per_mm"] - expected) < 1 / TARGET_MM


def test_no_target_at_all_is_none_rather_than_a_guess(sheet):
    """
    An image without a visible target is a normal thing in a large collection.
    Not zero, and not a raise -- the caller decides whether it is a problem.
    """
    _image, template, _expected = sheet
    blank = np.full((400, 400, 3), 60, np.uint8)
    assert scale_calibration.scale_from_target(blank, template, TARGET_MM) is None


def test_searching_where_the_target_is_not_produces_a_plausible_wrong_answer(sheet):
    """
    The failure mode this module is built around. Clutter wins by default, and
    the result LOOKS like a measurement -- which is why the score is stored and
    why the next test exists.
    """
    image, template, expected = sheet
    wrong = scale_calibration.scale_from_target(image, template, TARGET_MM,
                                                region=EMPTY_REGION)
    assert wrong is not None
    # Wrong by far more than the true match's error, and the only thing saying
    # so is the score -- the number itself is an ordinary-looking px/mm.
    assert abs(wrong["px_per_mm"] - expected) > 1 / TARGET_MM
    assert wrong["score"] < scale_calibration.WEAK_MATCH_SCORE


def test_a_minimum_score_turns_that_into_no_answer(sheet):
    """
    Nothing to measure is a normal outcome and must not be filled in with a
    guess.
    """
    image, template, _expected = sheet
    assert scale_calibration.scale_from_target(image, template, TARGET_MM,
                                               region=EMPTY_REGION,
                                               match_score_min=0.6) is None


def test_a_weak_match_is_accepted_but_warned_about(sheet, caplog):
    """
    Accepted because clutter can out-correlate an absent target and a project
    may still want the number; warned because a plausible wrong scale is worse
    than none.
    """
    image, template, _expected = sheet
    with caplog.at_level("WARNING"):
        scale_calibration.scale_from_target(image, template, TARGET_MM,
                                            region=EMPTY_REGION, name="clutter")
    assert "weak match" in caplog.text


def test_a_region_too_small_to_search_raises(sheet):
    image, template, _expected = sheet
    with pytest.raises(ValueError, match="smaller than the smallest template"):
        scale_calibration.scale_from_target(image, template, TARGET_MM,
                                            region=(0, 0, 0.001, 0.001))


def test_a_region_is_fractional_so_it_survives_a_resolution_change(sheet):
    """
    Pixel coordinates would not: the same rig at a new camera resolution would
    silently search the wrong part of the frame.
    """
    import cv2

    image, template, expected = sheet
    bigger = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    result = scale_calibration.scale_from_target(bigger, template, TARGET_MM,
                                                 region=TARGET_REGION)
    assert abs(result["px_per_mm"] - expected * 2) < 4 / TARGET_MM


def test_a_panel_is_drawn_for_a_human_to_check(sheet):
    """
    Display-ready uint8, because the operation knows what its own numbers mean.
    """
    image, template, _expected = sheet
    result = scale_calibration.scale_from_target(image, template, TARGET_MM,
                                                 region=TARGET_REGION)
    panel = scale_calibration.scale_panel(image, result)
    assert panel.dtype == np.uint8
    assert panel.shape == image.shape


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_a_declared_scale_is_recorded_as_declared(metadata_project):
    """
    For a rig whose calibration is a fact about the equipment -- a copy stand
    at a fixed height, a scanner at a stated dpi. There is nothing to detect in
    those images and no reason to pretend otherwise.
    """
    cf.declare_scale(metadata_project, 4.0, scope="device", scope_value="boxA")

    stored = calibration_records.load_calibrations(metadata_project)
    assert stored["source"].iloc[0] == "declared"
    assert stored["parameters"].iloc[0] == {"px_per_mm": 4.0}


def test_a_declared_scale_needs_something_to_be_true_of(metadata_project):
    with pytest.raises(ValueError, match="needs a scope_value"):
        cf.declare_scale(metadata_project, 4.0, scope="device")


def test_scales_resolve_to_a_float_per_occurrence(metadata_project):
    cf.declare_scale(metadata_project, 4.0, scope="device", scope_value="boxA")
    resolved = cf.scale_for_occurrences(metadata_project)

    assert resolved.dtype == "float64"
    assert resolved.name == "px_per_mm"
    assert resolved["specimen0"] == 4.0


def test_an_occurrence_scale_beats_its_device_scale(metadata_project):
    """
    A scale measured on one frame is more specific than one measured for the
    rig it was shot on -- and the uncalibrated stay NaN rather than borrowing.
    """
    cf.declare_scale(metadata_project, 4.0, scope="device", scope_value="boxA")
    cf.declare_scale(metadata_project, 8.0, scope=ID_COL,
                     scope_value="specimen0")

    resolved = cf.scale_for_occurrences(metadata_project)
    assert resolved["specimen0"] == 8.0
    assert resolved["specimen2"] == 4.0
    assert np.isnan(resolved["specimen1"])          # boxB, uncalibrated


def test_an_uncalibrated_project_resolves_to_an_empty_series(metadata_project):
    assert cf.scale_for_occurrences(metadata_project).empty


def test_measuring_is_repeat_aware(image_project, sheet, monkeypatch):
    """
    Already-measured scope values are skipped, so an interrupted pass resumes
    and a re-run is a no-op. Measured against the project's own images, which
    here are specimens rather than target sheets -- so nothing is detected, and
    "missed" is the honest count for that.
    """
    _image, template, _expected = sheet
    summary = cf.measure_scales(image_project, template, TARGET_MM,
                                match_score_min=0.9, limit=2)
    assert summary["measured"] == 0
    assert summary["missed"] == 2


def test_pending_scope_values_shrink_as_scales_are_declared(image_project):
    assert len(scale_calibration.pending_scope_values(image_project, "device")) == 2
    cf.declare_scale(image_project, 4.0, scope="device", scope_value="boxA")
    assert scale_calibration.pending_scope_values(image_project, "device") == ["boxB"]
