"""
Human labels as metrics, tested without a human.

A person's judgement is a derived value like any other: it stores in the metric
log, exports as a column, and is what automated QC gets calibrated against
(validation.filters). What makes these operations awkward to test is only the
window -- so the window is stubbed, on the module under test rather than on cv2
itself, and everything else is ordinary.

The stub replaces five names and forwards the rest to real cv2, which matters:
these functions genuinely call `cv2.circle` and `cv2.cvtColor` to build the
panel a person looks at, and a wholesale mock would make the assertions
meaningless. `waitKey` raises rather than blocking when it runs out of scripted
input, because the real loop is `while True` with no timeout and the alternative
to a loud failure is a suite that hangs forever.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics import annotation
from critterframe.metrics.annotation import (
    FLAG_KEYS,
    _point_pair,
    _skipped_pair,
)
from critterframe.recipes import Segment
from helpers.stubs import FakeCv2


def a_segment():
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True
    image = np.zeros((100, 100, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask, occurrence_id="specimen0")


@pytest.fixture
def gui(monkeypatch):
    """
    Script the window. Patched on `metrics.annotation`, never on cv2 -- these
    functions share cv2 with visualization.panels, whose output is being
    asserted on, and a global patch would leak into every other test.
    """
    def install(keys=(), clicks=()):
        fake = FakeCv2(keys=keys, clicks=clicks)
        monkeypatch.setattr(annotation, "cv2", fake)
        return fake
    return install


# ---------------------------------------------------------------------------
# The geometry, with no window at all
# ---------------------------------------------------------------------------


def test_two_points_give_a_length_and_a_direction():
    """A 3-4-5 triangle, so the answer is exact."""
    value = _point_pair(["head", "tail"], [(0, 0), (3, 4)])
    assert value["head"] == [0, 0]
    assert value["tail"] == [3, 4]
    assert value["length_px"] == 5.0


def test_the_angle_follows_image_coordinates():
    """
    y increases DOWNWARD, so a second point directly below the first is +90,
    not -90. This is the convention that is wrong for months before anyone
    notices, which is why it is pinned.
    """
    assert _point_pair(["a", "b"], [(0, 0), (0, 10)])["angle_deg"] == 90.0
    assert _point_pair(["a", "b"], [(0, 0), (0, -10)])["angle_deg"] == -90.0
    assert _point_pair(["a", "b"], [(0, 0), (10, 0)])["angle_deg"] == 0.0


def test_both_raw_points_are_kept():
    """
    A length and an angle are each recoverable from two points; neither
    recovers the points, and which comparison you will want isn't knowable at
    annotation time.
    """
    value = _point_pair(["head", "tail"], [(12, 34), (56, 78)])
    assert value["head"] == [12, 34] and value["tail"] == [56, 78]


def test_a_skipped_occurrence_keeps_the_shape_with_nothing_in_it():
    """
    "Looked at and passed over" is a different fact from "never reached", and
    only the first is recoverable from a stored value.
    """
    assert _skipped_pair(["head", "tail"]) == {
        "head": None, "tail": None, "length_px": None, "angle_deg": None}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_four_flags_are_the_reasons_worth_telling_apart():
    """
    A metric that catches every cut-off organism while missing every
    non-organism is a different instrument from one aggregate "bad" rate.
    """
    assert set(FLAG_KEYS.values()) == {"usable", "not_an_organism", "cut_off",
                                       "multiple_organisms"}


def test_point_labels_must_be_two_and_distinct():
    """Validated before any window opens, so a typo fails immediately."""
    for labels in (["head"], ["head", "tail", "wing"], ["head", "head"]):
        with pytest.raises(ValueError, match="two distinct labels"):
            cf.click_two_points(labels)


def test_the_labels_are_part_of_the_recipe():
    """
    They name what was clicked, so two projects clicking different things are
    not doing the same work under one name.
    """
    assert (cf.click_two_points(["head", "tail"]).spec()
            != cf.click_two_points(["base", "tip"]).spec())


def test_a_label_metric_is_a_category_not_a_measurement():
    assert cf.annotate_flags().unit == "category"
    assert cf.click_two_points().unit == "px_xy"


def test_click_units_do_not_convert_to_millimetres():
    """
    "px_xy" is deliberately not one of the convertible units: a label whose job
    is grading a pipeline measured in pixels should stay in pixels, and the key
    names carry their own units so the coarse parent tag can't mislead.
    """
    from critterframe.export import CONVERTIBLE_UNITS

    assert cf.click_two_points().unit not in CONVERTIBLE_UNITS


# ---------------------------------------------------------------------------
# The window, scripted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key, expected", sorted(FLAG_KEYS.items()))
def test_each_key_records_its_flag(gui, key, expected):
    fake = gui(keys=[key])
    assert cf.annotate_flags()(a_segment()) == expected
    assert fake.shown, "the annotator was never shown anything"


def test_a_key_that_means_nothing_is_ignored_rather_than_recorded(gui):
    """
    A mis-key must not be stored as a judgement -- the loop waits for one of
    the valid keys.
    """
    gui(keys=[ord("z"), ord("q"), ord("1")])
    assert cf.annotate_flags()(a_segment()) == "usable"


def test_the_window_is_closed_afterwards(gui):
    fake = gui(keys=[ord("1")])
    cf.annotate_flags()(a_segment())
    assert fake.destroyed


def test_two_clicks_become_a_measurement(gui):
    """
    The whole plumbing: the mouse callback collects points, and the geometry
    tested above turns them into the stored value.
    """
    # Three keys for two clicks: the loop polls once per click, and there is a
    # third, cosmetic wait that holds the second marker on screen briefly.
    gui(keys=[ord(" ")] * 3,
        clicks=[(cv2.EVENT_LBUTTONDOWN, 10, 20), (cv2.EVENT_LBUTTONDOWN, 40, 60)])

    value = cf.click_two_points()(a_segment())
    assert value["head"] == [10, 20]
    assert value["tail"] == [40, 60]
    assert value["length_px"] == pytest.approx(50.0)


def test_escape_skips_the_occurrence(gui):
    """
    For an occurrence where one of the points isn't visible -- which is a
    label, not a gap.
    """
    gui(keys=[27])
    value = cf.click_two_points()(a_segment())
    assert value == {"head": None, "tail": None, "length_px": None,
                     "angle_deg": None}


def test_clicking_needs_a_mask_to_show(gui):
    """The overlay is what the person is clicking on."""
    gui(keys=[27])
    with pytest.raises(ValueError, match="has no mask yet"):
        cf.click_two_points()(Segment(np.zeros((10, 10, 3), np.uint8)))


def test_a_stub_that_runs_dry_fails_instead_of_hanging(gui):
    """
    The real loop is `while True: waitKey(20)` with no timeout. A test whose
    script is incomplete has to fail, not wait forever.
    """
    gui(keys=[])
    with pytest.raises(AssertionError, match="ran dry"):
        cf.annotate_flags()(a_segment())


@pytest.mark.slow
def test_labels_run_and_store_like_any_other_metric(gui, segmented_project):
    """
    The point of labels being metrics: one recipe, one run record, one export
    column, and repeat-awareness -- so an interrupted annotation session
    resumes where the person stopped rather than asking them again.
    """
    gui(keys=[ord("1")] * 8)
    first = cf.run_metrics(segmented_project, run_name="screening",
                           metrics=[cf.annotate_flags()],
                           visualize=False)["organism"]
    assert first["processed"] == 8

    gui(keys=[])            # a second pass must ask nobody anything
    second = cf.run_metrics(segmented_project, run_name="screening",
                            metrics=[cf.annotate_flags()],
                            visualize=False)["organism"]
    assert second["skipped"] == 8

    exported = cf.export_metrics(segmented_project, runs=["screening"])
    assert set(exported["screening__organism__annotate_flags"]) == {"usable"}
