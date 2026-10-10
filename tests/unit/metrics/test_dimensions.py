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
from critterframe.core.recipes import Segment


def rectangle_mask(width=40, height=100, shape=(200, 200)):
    mask = np.zeros(shape, bool)
    top = (shape[0] - height) // 2
    left = (shape[1] - width) // 2
    mask[top : top + height, left : left + width] = True
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


@pytest.mark.parametrize("metric", [cf.body_length(), cf.max_width(), cf.mask_area(), cf.bounding_box()])
def test_every_dimension_refuses_an_empty_mask(metric):
    """
    Zero would be a measurement. An empty mask is a failed segmentation, and
    the run counts it as a failure rather than exporting a specimen 0 px long.
    """
    with pytest.raises(ValueError, match="empty mask"):
        metric(a_segment(np.zeros((50, 50), bool)))


@pytest.mark.parametrize("metric", [cf.body_length(), cf.max_width(), cf.mask_area(), cf.bounding_box()])
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
    cf.run_metrics(measured_project, run_name="boxes", metrics=[cf.bounding_box()], visualize=False)
    exported = cf.export_metrics(measured_project, run_names=["boxes"])
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


@pytest.mark.parametrize(
    "metric, stage",
    [
        (cf.body_length(), "body_length"),
        (cf.max_width(), "max_width"),
        (cf.mask_area(), "mask_area"),
    ],
)
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


# ---------------------------------------------------------------------------
# elongation
# ---------------------------------------------------------------------------


def ellipse_mask(semi_long=60, semi_short=15, angle=0, shape=(240, 240)):
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (shape[1] // 2, shape[0] // 2), (semi_long, semi_short), angle, 0, 360, 1, -1)
    return mask.astype(bool)


def test_a_rectangle_is_as_elongated_as_its_sides_say():
    """Exact for a rectangle: its variances are L²/12 and W²/12."""
    assert cf.elongation()(a_segment(rectangle_mask(width=20, height=100))) == pytest.approx(5.0, rel=0.01)


def test_a_one_pixel_wide_line_still_has_a_width():
    """
    A pixel is a square, not a point. Counted as points, a line one pixel wide
    has no width at all and the ratio is infinite.
    """
    assert cf.elongation()(a_segment(rectangle_mask(width=1, height=100))) == pytest.approx(100, rel=0.01)


def test_an_ellipse_is_as_elongated_as_its_axes():
    """cv2 draws the boundary pixel too, so semi-axes of 60 and 15 cover 60.5 and 15.5."""
    assert cf.elongation()(a_segment(ellipse_mask(60, 15))) == pytest.approx(60.5 / 15.5, rel=0.01)
    assert cf.elongation()(a_segment(ellipse_mask(40, 40))) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("angle", [30, 75])
def test_it_needs_no_orientation(angle):
    """It finds the mask's own axes, so a rotated mask measures the same."""
    level = cf.elongation()(a_segment(ellipse_mask(60, 15)))
    assert cf.elongation()(a_segment(ellipse_mask(60, 15, angle))) == pytest.approx(level, rel=0.02)


def test_pixels_far_from_the_body_pull_it_toward_round():
    """
    Variance weighs a pixel by its squared distance from the centre, so a long
    thin leg or a far speck counts for much more than its area. That is the
    signal where a mask has leaked, and noise where the body's own shape is
    wanted, which is what removing islands and appendages first is for.
    """
    body = ellipse_mask(60, 15, angle=90)  # long axis vertical
    leg = body.copy()
    leg[118:121, 135:215] = True  # 80px sideways, 3px thick
    speck = body.copy()
    speck[10:13, 10:13] = True  # nine pixels, far away

    plain = cf.elongation()(a_segment(body))
    assert cf.elongation()(a_segment(leg)) < 0.5 * plain
    assert cf.elongation()(a_segment(speck)) < 0.9 * plain

    cleaned, _info = cf.remove_islands()(a_segment(speck))
    assert cf.elongation()(cleaned) == pytest.approx(plain)

    # Still less swayed than the widest row, which takes the leg's whole length.
    by_rows = cf.body_length()(a_segment(leg)) / cf.max_width()(a_segment(leg))
    assert cf.elongation()(a_segment(leg)) > by_rows


def test_a_mask_with_no_shape_is_refused():
    one_pixel = np.zeros((10, 10), bool)
    one_pixel[5, 5] = True
    with pytest.raises(ValueError, match="too few pixels"):
        cf.elongation()(a_segment(one_pixel))
    with pytest.raises(ValueError, match="too few pixels"):
        cf.elongation()(a_segment(np.zeros((10, 10), bool)))


def test_elongation_is_a_ratio_and_does_not_convert():
    from critterframe.wide import CONVERTIBLE_UNITS

    assert cf.elongation().unit == "ratio"
    assert "ratio" not in CONVERTIBLE_UNITS
    assert cf.elongation(name="abdomen_elongation").metric_name == "abdomen_elongation"
    assert cf.elongation().spec() == cf.elongation().spec()


def test_the_panel_draws_the_axes_it_measured():
    class Sink:
        def __init__(self):
            self.panels = []

        def collect(self, occurrence_id, stage, image):
            self.panels.append((stage, image))

    segment = a_segment(ellipse_mask(60, 15, angle=30))
    cf.elongation()(segment)  # no sink, nothing drawn

    segment.panel_sink = Sink()
    cf.elongation()(segment)
    ((stage, panel),) = segment.panel_sink.panels
    assert stage == "elongation"
    assert ((panel == (0, 255, 0)).all(axis=2)).any()  # the long axis
    assert ((panel == (0, 0, 255)).all(axis=2)).any()  # the short axis


# ---------------------------------------------------------------------------
# jaggedness
# ---------------------------------------------------------------------------


def toothed(mask, teeth, depth=3):
    """`mask` with `teeth` notches of radius `depth` bitten out of its edge, evenly spaced."""
    bitten = mask.astype(np.uint8)
    contours, _hierarchy = cv2.findContours(bitten.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = contours[0][:, 0, :]
    for index in range(0, len(outline), max(1, len(outline) // teeth)):
        cv2.circle(bitten, tuple(int(v) for v in outline[index]), depth, 0, -1)
    return bitten.astype(bool)


@pytest.mark.parametrize(
    "mask",
    [
        ellipse_mask(20, 20),
        ellipse_mask(60, 60),
        ellipse_mask(100, 25),
        rectangle_mask(width=40, height=150),
        ellipse_mask(60, 15, angle=30),
    ],
    ids=["small disc", "large disc", "long ellipse", "rectangle", "rotated ellipse"],
)
def test_a_smooth_mask_reads_smooth_at_any_size(mask):
    """A raster edge's staircase is on both sides of the ratio and cancels."""
    assert cf.jaggedness()(a_segment(mask)) == pytest.approx(1.0, abs=0.05)


def test_a_smooth_curve_reads_smooth_where_solidity_would_not():
    """The measure is about the edge, not the shape: a bent part is not a rough one."""
    arc = np.zeros((300, 300), np.uint8)
    cv2.ellipse(arc, (150, 200), (110, 110), 0, 200, 340, 1, 30)
    assert cf.jaggedness()(a_segment(arc.astype(bool))) == pytest.approx(1.0, abs=0.05)


def test_a_toothed_edge_reads_jagged_and_more_teeth_read_more_jagged():
    body = ellipse_mask(100, 30, shape=(300, 300))
    values = [cf.jaggedness()(a_segment(toothed(body, teeth))) for teeth in (10, 30, 60)]

    assert values[0] > cf.jaggedness()(a_segment(body)) + 0.05
    assert values == sorted(values) and values[-1] > 1.5


def test_jaggedness_needs_no_orientation():
    upright = toothed(ellipse_mask(80, 25, shape=(260, 260)), 30)
    turned = cv2.warpAffine(
        upright.astype(np.uint8),
        cv2.getRotationMatrix2D((130, 130), 40, 1.0),
        (260, 260),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    assert cf.jaggedness()(a_segment(turned)) == pytest.approx(cf.jaggedness()(a_segment(upright)), abs=0.08)


def test_a_leg_counts_as_roughness_until_it_is_removed():
    """
    A thin appendage is narrower than the smoothing scale, so the smoothed
    outline goes without it. That is the right answer for a quality filter, and
    remove_appendages() first is the answer for the body's own edge.
    """
    body = ellipse_mask(100, 30, shape=(300, 300))
    leg = body.copy()
    leg[148:151, 250:290] = True

    assert cf.jaggedness()(a_segment(leg)) > cf.jaggedness()(a_segment(body)) + 0.1
    cleaned, _info = cf.remove_appendages()(a_segment(leg))
    assert cf.jaggedness()(cleaned) == pytest.approx(cf.jaggedness()(a_segment(body)), abs=0.05)


def test_the_smoothing_scale_is_part_of_the_recipe():
    assert cf.jaggedness().spec() == cf.jaggedness(0.2).spec()
    assert cf.jaggedness().spec() != cf.jaggedness(0.4).spec()
    assert cf.jaggedness().spec() != cf.jaggedness(px=3).spec()
    assert cf.jaggedness().unit == "ratio"


def test_a_coarser_scale_counts_coarser_irregularity():
    """Notches the size of the scale are smoothed away at a coarse scale and kept at a fine one."""
    rough = toothed(ellipse_mask(100, 30, shape=(300, 300)), 20, depth=6)
    assert cf.jaggedness(px=10)(a_segment(rough)) > cf.jaggedness(px=1)(a_segment(rough))


@pytest.mark.parametrize("kwargs", [{"fraction": 0}, {"fraction": 1}, {"px": 0}, {"px": -1}])
def test_a_smoothing_scale_that_makes_no_sense_fails_before_it_runs(kwargs):
    with pytest.raises(ValueError, match="must be"):
        cf.jaggedness(**kwargs)


def test_a_mask_with_no_edge_to_speak_of_is_refused():
    one_pixel = np.zeros((10, 10), bool)
    one_pixel[5, 5] = True
    with pytest.raises(ValueError, match="too few pixels"):
        cf.jaggedness()(a_segment(one_pixel))

    speck = np.zeros((40, 40), bool)
    speck[20:22, 20:22] = True
    with pytest.raises(ValueError, match="too small for the smoothing scale"):
        cf.jaggedness(px=10)(a_segment(speck))


def test_the_jaggedness_panel_draws_the_smoothed_outline():
    class Sink:
        def __init__(self):
            self.panels = []

        def collect(self, occurrence_id, stage, image):
            self.panels.append((stage, image))

    segment = a_segment(toothed(ellipse_mask(100, 30, shape=(300, 300)), 30))
    cf.jaggedness()(segment)  # no sink, nothing drawn

    segment.panel_sink = Sink()
    cf.jaggedness()(segment)
    ((stage, panel),) = segment.panel_sink.panels
    assert stage == "jaggedness"
    assert ((panel == (0, 0, 255)).all(axis=2)).any()
