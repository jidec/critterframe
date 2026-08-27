"""
Where the organism is -- reported in ORIGINAL image coordinates, always.

Position is the one family of metrics whose answer would be meaningless in the
working frame. After `crop_to_mask()` every specimen is at the centre of its own
crop, so a centroid measured there would be the same number for every occurrence
in the project. These metrics therefore invert the affine before measuring, and
that is the property under test.

What position is FOR is the questions a trait table can't otherwise ask: where
on the light-trap sheet the moths landed, whether the specimens a collection
photographed drift toward one corner, whether "cut off" correlates with a
position in the frame.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment

FRAME = (200, 400)          # height, width
BOX = (slice(40, 60), slice(300, 340))      # y, x -- right of centre, high up


def a_segment():
    image = np.zeros((*FRAME, 3), np.uint8)
    image[BOX] = 255
    mask = np.zeros(FRAME, bool)
    mask[BOX] = True
    return Segment(image, mask=mask, occurrence_id="test")


def cropped_and_rotated():
    """A working frame that has moved a long way from the original."""
    segment = a_segment()
    for operation in (cf.crop_to_mask(pad=0.1), cf.rotate(20),
                      cf.resize(scale=2.0)):
        segment, _info = operation(segment)
    return segment


# ---------------------------------------------------------------------------
# centroid
# ---------------------------------------------------------------------------


def test_the_centroid_is_where_the_organism_is_in_the_photograph():
    centre = cf.centroid()(a_segment())
    assert centre["x"] == pytest.approx(319.5, abs=1)
    assert centre["y"] == pytest.approx(49.5, abs=1)


def test_the_centroid_survives_a_crop_and_a_rotation():
    """
    THE property. Without the inversion, every cropped specimen would report
    the centre of its own crop and the column would carry no information at all.
    """
    original = cf.centroid()(a_segment())
    transformed = cf.centroid()(cropped_and_rotated())

    assert transformed["x"] == pytest.approx(original["x"], abs=4)
    assert transformed["y"] == pytest.approx(original["y"], abs=4)


# ---------------------------------------------------------------------------
# relative_position
# ---------------------------------------------------------------------------


def test_relative_position_is_a_fraction_of_the_frame():
    """
    Fractions rather than pixels, so positions from cameras of different
    resolutions are comparable -- which is the only way the question "do
    specimens drift toward one corner" survives a hardware change.
    """
    position = cf.relative_position()(a_segment())
    assert position["x"] == pytest.approx(319.5 / 400, abs=0.01)
    assert position["y"] == pytest.approx(49.5 / 200, abs=0.01)


def test_relative_position_survives_the_working_frame_changing():
    original = cf.relative_position()(a_segment())
    transformed = cf.relative_position()(cropped_and_rotated())
    assert transformed["x"] == pytest.approx(original["x"], abs=0.02)
    assert transformed["y"] == pytest.approx(original["y"], abs=0.02)


def test_relative_position_stays_inside_the_frame():
    position = cf.relative_position()(a_segment())
    assert 0 <= position["x"] <= 1 and 0 <= position["y"] <= 1


# ---------------------------------------------------------------------------
# image_bounds
# ---------------------------------------------------------------------------


def test_image_bounds_are_the_box_in_the_original_photograph():
    bounds = cf.image_bounds()(a_segment())
    assert bounds == {"x": 300, "y": 40, "width": 40, "height": 20}


def test_image_bounds_survive_a_crop():
    """
    Which is what makes them joinable back to the photograph: a figure that
    draws these boxes on the original sheet needs them in the sheet's
    coordinates.
    """
    bounds = cf.image_bounds()(cropped_and_rotated())
    assert bounds["x"] == pytest.approx(300, abs=8)
    assert bounds["y"] == pytest.approx(40, abs=8)


# ---------------------------------------------------------------------------
# Shared contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric, unit", [
    (cf.centroid(), "px"),
    (cf.relative_position(), "fraction"),
    (cf.image_bounds(), "px"),
])
def test_units_distinguish_pixels_from_fractions(metric, unit):
    """
    Which is what tells the export whether a column can be converted to
    millimetres: a fraction has no length in it.
    """
    assert metric.unit == unit


@pytest.mark.parametrize("metric", [cf.centroid(), cf.relative_position(),
                                    cf.image_bounds()])
def test_every_position_metric_needs_a_mask(metric):
    with pytest.raises(ValueError, match="has no mask yet"):
        metric(Segment(np.zeros((*FRAME, 3), np.uint8)))


@pytest.mark.parametrize("metric", [cf.centroid(), cf.relative_position(),
                                    cf.image_bounds()])
def test_every_position_metric_reports_several_numbers_at_once(metric):
    """
    A position is not one number, and splitting it into two operations would
    invert the coordinate chain twice for one answer.
    """
    assert isinstance(metric(a_segment()), dict)


def test_positions_export_as_one_column_per_key(measured_project):
    cf.run_metrics(measured_project, run_name="where",
                   metrics=[cf.centroid(), cf.relative_position()],
                   visualize=False)
    exported = cf.export_metrics(measured_project, runs=["where"])

    assert "where__organism__centroid__x" in exported.columns
    assert "where__organism__relative_position__y" in exported.columns
