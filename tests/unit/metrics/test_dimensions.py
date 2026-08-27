"""
Size traits, measured on shapes whose size is known exactly.

Every measurement here is taken in the segment's CURRENT frame, which is what
makes `orient()` before measuring meaningful: body_length is the vertical extent,
so it means "length" only once the body is vertical. That is a deliberate
division of labour -- the transform decides what "along the body" means, the
metric just measures it -- and it is why these tests draw shapes at a known
angle rather than trusting the metric to find one.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment


def rectangle_mask(width=40, height=100, shape=(200, 200)):
    mask = np.zeros(shape, bool)
    top = (shape[0] - height) // 2
    left = (shape[1] - width) // 2
    mask[top:top + height, left:left + width] = True
    return mask


def a_segment(mask=None, image=None):
    mask = rectangle_mask() if mask is None else mask
    if image is None:
        image = np.zeros((*mask.shape, 3), np.uint8)
        image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="test")


# ---------------------------------------------------------------------------
# body_length / max_width / mask_area
# ---------------------------------------------------------------------------


def test_body_length_is_the_vertical_extent_to_the_pixel():
    assert cf.body_length()(a_segment(rectangle_mask(height=100))) == 100


def test_max_width_is_the_widest_row_to_the_pixel():
    assert cf.max_width()(a_segment(rectangle_mask(width=40))) == 40


def test_mask_area_is_the_pixel_count():
    assert cf.mask_area()(a_segment(rectangle_mask(40, 100))) == 4000


def test_max_width_finds_a_bulge_rather_than_averaging():
    """
    An extremum, not a mean: the widest point of a thorax is a real trait, and
    a mean width would report something no part of the specimen measures.
    """
    mask = rectangle_mask(width=20, height=100)
    mask[100:104, 60:140] = True
    assert cf.max_width()(a_segment(mask)) == 80


def test_length_measures_the_frame_it_is_given():
    """
    Which is why a recipe orients first. Rotate the specimen and "length"
    becomes the width -- the metric is not wrong, it is measuring what it was
    handed.
    """
    upright = rectangle_mask(width=40, height=100)
    sideways = np.rot90(upright).copy()

    assert cf.body_length()(a_segment(upright)) == 100
    assert cf.body_length()(a_segment(sideways)) == 40


def test_a_disconnected_speck_still_counts_toward_the_extent():
    """
    The honest behaviour for a mask that has one: length is an extent, not a
    body. Cleaning that up is remove_appendages' job, done before measuring.
    """
    mask = rectangle_mask(height=100)
    mask[10, 100] = True
    assert cf.body_length()(a_segment(mask)) > 100


@pytest.mark.parametrize("metric", [cf.body_length(), cf.max_width(),
                                    cf.mask_area(), cf.bounding_box()])
def test_every_dimension_refuses_an_empty_mask(metric):
    """
    Zero would be a measurement. An empty mask is a failed segmentation, and
    the run counts it as a failure rather than exporting a specimen 0 px long.
    """
    with pytest.raises(ValueError, match="empty mask"):
        metric(a_segment(np.zeros((50, 50), bool)))


@pytest.mark.parametrize("metric", [cf.body_length(), cf.max_width(),
                                    cf.mask_area(), cf.bounding_box()])
def test_every_dimension_needs_a_mask_at_all(metric):
    with pytest.raises(ValueError, match="has no mask yet"):
        metric(Segment(np.zeros((50, 50, 3), np.uint8)))


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------


def test_a_bounding_box_reports_all_four_numbers():
    box = cf.bounding_box()(a_segment(rectangle_mask(40, 100)))
    assert box == {"x": 80, "y": 50, "width": 40, "height": 100}


def test_a_dict_valued_metric_becomes_one_column_per_key(measured_project):
    """
    Which is the reason a metric is allowed to return several numbers at once
    rather than being split into four operations that each re-derive the mask.
    """
    cf.run_metrics(measured_project, run_name="boxes",
                   metrics=[cf.bounding_box()], visualize=False)
    exported = cf.export_metrics(measured_project, runs=["boxes"])
    assert "boxes__organism__bounding_box__width" in exported.columns


# ---------------------------------------------------------------------------
# Naming and units
# ---------------------------------------------------------------------------


def test_a_metric_can_be_stored_under_another_name():
    """
    So the same operation can appear twice in one recipe under different
    configurations without the second overwriting the first.
    """
    metric = cf.mask_area(name="area_px")
    assert metric.metric_name == "area_px"
    assert metric.name == "mask_area"


def test_the_unit_travels_with_the_value():
    assert cf.body_length().unit == "px"
    assert cf.mask_area().unit == "px2"
    assert cf.mask_area(unit="mm2").unit == "mm2"


def test_length_is_body_length_under_another_name():
    """A convenience alias, and the same operation -- so it hashes the same."""
    assert cf.length is cf.body_length


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.panels = []

    def collect(self, occurrence_id, stage, image):
        self.panels.append((stage, image))


@pytest.mark.parametrize("metric, stage", [
    (cf.body_length(), "body_length"),
    (cf.max_width(), "max_width"),
    (cf.mask_area(), "mask_area"),
])
def test_a_measurement_draws_what_it_measured(metric, stage):
    """
    A number is not checkable by eye; a line drawn across the specimen is. The
    panel is display-ready uint8, because the operation is what knows what its
    own numbers mean.
    """
    sink = RecordingSink()
    mask = rectangle_mask()
    image = np.zeros((*mask.shape, 3), np.uint8)
    metric(Segment(image, mask=mask, occurrence_id="test", panel_sink=sink))

    assert [name for name, _ in sink.panels] == [stage]
    assert sink.panels[0][1].dtype == np.uint8


def test_no_panel_is_built_when_nobody_is_listening():
    """
    The great majority of occurrences run with no sink, and drawing for them
    would be pure waste -- which is why each metric checks before rendering.
    """
    mask = rectangle_mask()
    image = np.zeros((*mask.shape, 3), np.uint8)
    assert cf.body_length()(Segment(image, mask=mask)) == 100


def test_a_drawn_specimen_measures_close_to_what_was_drawn(draw_specimen):
    """
    One end-to-end sanity check against the shared synthetic specimen, so a
    change in the drawing helper cannot silently invalidate every metric test
    that uses it.
    """
    image = draw_specimen(0, legs=False)
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 100
    segment = Segment(image, mask=mask, occurrence_id="test")

    oriented, _info = cf.orient()(segment)
    assert cf.body_length()(oriented) == pytest.approx(120, abs=8)
    assert cf.max_width()(oriented) == pytest.approx(40, abs=8)
