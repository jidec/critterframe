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
    LABEL_KEYS,
    _point_pair,
    _skipped_pair,
)
from critterframe.recipes import Segment
from critterframe.visualization import panels
from helpers.stubs import FakeCv2


# A project's own screening vocabulary: a label about the image, asked before any mask exists.
SCREEN = ["usable", "cut_off", "dead"]


def screen():
    return cf.exclusive_label_annotation(SCREEN, name="usability", requires_mask=False)


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
        monkeypatch.setattr(panels, "DISPLAY_MAX", None)   # 1:1 window, so clicks are image pixels
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


def test_click_two_points_still_requires_a_mask():
    """It clicks points ON the segment, so unlike a label about the image
    there's nothing to click without one."""
    assert cf.click_two_points().requires_mask is True


def test_the_panel_falls_back_to_the_image_alone_without_a_mask():
    from critterframe.metrics.annotation import _panel

    image = np.zeros((10, 10, 3), np.uint8)
    panel = _panel(Segment(image, occurrence_id="x"))
    assert panel.shape == image.shape


def test_the_panel_is_the_wider_side_by_side_view_with_a_mask():
    from critterframe.metrics.annotation import _panel

    panel = _panel(a_segment())
    # side_by_side of three 100x100 panels is much wider than any one of them.
    assert panel.shape[1] > 100 * 2


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
    assert screen().unit == "category"
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


def test_it_works_end_to_end_on_a_segment_with_no_mask(gui):
    """The real point of requires_mask=False: this has to be usable on a
    fresh, unsegmented occurrence, not just tolerate one in theory."""
    fake = gui(keys=[ord("3")])   # 3 = dead
    image = np.zeros((10, 10, 3), np.uint8)
    assert screen()(Segment(image, occurrence_id="x")) == "dead"
    assert fake.shown


QUALITY = ["good", "input_invalid", "wrong_region", "incomplete", "overflow"]


@pytest.mark.parametrize("key, expected", list(zip("12345", QUALITY)))
def test_each_key_records_its_label_in_list_order(gui, key, expected):
    """The vocabulary is the caller's; the keys are handed out in the order given."""
    fake = gui(keys=[ord(key)])
    assert cf.exclusive_label_annotation(QUALITY)(a_segment()) == expected
    assert fake.shown, "the annotator was never shown anything"


def test_the_legend_is_the_callers_vocabulary(gui):
    """So a key that belongs to another vocabulary means nothing here and is ignored."""
    from critterframe.metrics.annotation import _keys_for, _legend_lines

    shown = " ".join(_legend_lines(_keys_for(QUALITY))).split()
    assert [entry.split("=", 1)[1] for entry in shown] == QUALITY

    gui(keys=[ord("9"), ord("c"), ord("4")])     # two keys of a longer vocabulary, then a real one
    assert cf.exclusive_label_annotation(QUALITY)(a_segment()) == "incomplete"


def test_keys_run_past_the_digits_into_letters(gui):
    labels = [f"label{index}" for index in range(12)]
    gui(keys=[ord("0")])
    assert cf.exclusive_label_annotation(labels)(a_segment()) == "label9"
    gui(keys=[ord("b")])
    assert cf.exclusive_label_annotation(labels)(a_segment()) == "label11"


def test_the_labels_are_part_of_what_was_asked():
    """
    A different vocabulary is a different question, so it is a different
    recipe: labels given under one list must not read as answers to another.
    """
    base = cf.exclusive_label_annotation(QUALITY)
    assert base.spec() == cf.exclusive_label_annotation(list(QUALITY)).spec()
    assert base.spec() != cf.exclusive_label_annotation(QUALITY + ["smudged"]).spec()
    assert base.spec() != cf.exclusive_label_annotation(QUALITY[::-1]).spec()


def test_whether_a_label_needs_a_mask_is_the_callers_and_not_in_the_recipe(gui):
    """
    A label that describes the mask is asked only where there is one; a label
    that describes the image is asked of everything. Which occurrences a run
    reaches is not what one occurrence's label is, so it stays out of the hash.
    """
    of_the_mask = cf.exclusive_label_annotation(QUALITY)
    of_the_image = cf.exclusive_label_annotation(QUALITY, requires_mask=False)

    assert of_the_mask.requires_mask is True and of_the_image.requires_mask is False
    assert of_the_mask.spec() == of_the_image.spec()

    maskless = Segment(np.zeros((10, 10, 3), np.uint8), occurrence_id="x")
    gui(keys=[ord("1")])
    with pytest.raises(ValueError, match="has no mask yet"):
        of_the_mask(maskless)
    assert of_the_image(maskless) == "good"


@pytest.mark.parametrize("labels, message", [
    ([], "at least one label"),
    (["good", "bad", "good"], "distinct labels"),
    ([f"label{index}" for index in range(len(LABEL_KEYS) + 1)], "keys to hand out"),
])
def test_a_vocabulary_that_cannot_be_asked_fails_before_any_window_opens(labels, message):
    with pytest.raises(ValueError, match=message):
        cf.exclusive_label_annotation(labels)


def test_the_name_is_what_the_label_is_stored_under():
    assert cf.exclusive_label_annotation(QUALITY).metric_name == "exclusive_label_annotation"
    assert cf.exclusive_label_annotation(QUALITY, name="abdomen_quality").metric_name == "abdomen_quality"


def test_a_key_that_means_nothing_is_ignored_rather_than_recorded(gui):
    """
    A mis-key must not be stored as a judgement -- the loop waits for one of
    the valid keys.
    """
    gui(keys=[ord("z"), ord("q"), ord("1")])
    assert screen()(a_segment()) == "usable"


def test_the_window_is_closed_afterwards(gui):
    fake = gui(keys=[ord("1")])
    screen()(a_segment())
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
        screen()(a_segment())


@pytest.mark.slow
def test_labels_run_and_store_like_any_other_metric(gui, segmented_project):
    """
    The point of labels being metrics: one recipe, one run record, one export
    column, and repeat-awareness -- so an interrupted annotation session
    resumes where the person stopped rather than asking them again.
    """
    gui(keys=[ord("1")] * 8)
    first = cf.run_metrics(segmented_project, run_name="screening",
                           metrics=[screen()],
                           visualize=False)["organism"]
    assert first["processed"] == 8

    gui(keys=[])            # a second pass must ask nobody anything
    second = cf.run_metrics(segmented_project, run_name="screening",
                            metrics=[screen()],
                            visualize=False)["organism"]
    assert second["skipped"] == 8

    exported = cf.export_metrics(segmented_project, run_names=["screening"])
    assert set(exported["screening__organism__usability"]) == {"usable"}


# ---------------------------------------------------------------------------
# A window fitted to the screen
# ---------------------------------------------------------------------------


def test_clicked_points_come_back_in_segment_pixels_not_window_pixels(gui, monkeypatch):
    gui(keys=[ord(" ")] * 3,
        clicks=[(cv2.EVENT_LBUTTONDOWN, 20, 40), (cv2.EVENT_LBUTTONDOWN, 80, 120)])
    monkeypatch.setattr(panels, "DISPLAY_MAX", (200, 200))   # 100px shown 2x
    value = cf.click_two_points()(a_segment())

    assert value["head"] == [10, 20]
    assert value["tail"] == [40, 60]


def test_the_label_panel_fits_the_screen_box(gui, monkeypatch):
    fake = gui(keys=[ord("1")])
    monkeypatch.setattr(panels, "DISPLAY_MAX", (150, 150))   # three 100px panels, 300 wide
    segment = a_segment()
    before = segment.image.copy()
    screen()(segment)

    height, width = fake.shown[-1][1].shape[:2]
    assert width <= 150 and height <= 150
    assert np.array_equal(segment.image, before)


# ---------------------------------------------------------------------------
# The procedure behind a label
# ---------------------------------------------------------------------------


def test_a_labelling_note_does_not_move_the_recipe():
    """
    What a label MEANS is the vocabulary, and that is hashed. How the annotator
    was told to apply it is a note: reword it and the labels already given are
    still answers to the same question.
    """
    labels = ["good", "incomplete"]
    plain = cf.exclusive_label_annotation(labels)
    noted = cf.exclusive_label_annotation(labels, note="incomplete: a segment or more missing")
    reworded = cf.exclusive_label_annotation(labels, note="incomplete = at least one segment gone")

    assert plain.spec() == noted.spec() == reworded.spec()
    assert noted.note == "incomplete: a segment or more missing"
    assert noted.prepare(None) == {"note": "incomplete: a segment or more missing"}
    assert plain.prepare(None) is None


# ---------------------------------------------------------------------------
# The original image beside a transformed segment
# ---------------------------------------------------------------------------


def a_cropped_segment():
    """A specimen that is a small part of a larger photo, cropped down to it."""
    image = np.full((200, 300, 3), 40, np.uint8)
    mask = np.zeros((200, 300), bool)
    mask[60:160, 200:230] = True
    image[mask] = 200
    segment, _info = cf.crop_to_mask()(Segment(image, mask=mask, occurrence_id="specimen0"))
    return segment, image


def test_the_original_image_can_come_first(gui):
    """
    A crop with its background removed shows the part and nothing around it.
    The untouched photo beside it is the context a label is often judged on.
    """
    segment, image = a_cropped_segment()
    labels = ["good", "poor"]

    from critterframe.metrics.annotation import _panel
    from critterframe.visualization.panels import DISPLAY_MAX

    with_original = gui(keys=[ord("2")])
    assert cf.exclusive_label_annotation(labels, show_original=True)(segment) == "poor"

    shown = with_original.shown[-1][1]
    assert shown.shape[1] <= DISPLAY_MAX[0] and shown.shape[0] <= DISPLAY_MAX[1]
    # the original and the working views share one height, side by side
    working = _panel(segment)
    original_width = int(image.shape[1] * shown.shape[0] / image.shape[0])
    working_width = shown.shape[1] - original_width
    assert working_width / shown.shape[0] == pytest.approx(
        working.shape[1] / working.shape[0], rel=0.02)


def test_the_original_is_not_shrunk_to_the_crop_before_it_is_shown(gui):
    """
    A small part of a large photo: the photo is resized once, to the height it
    is shown at, so detail finer than the crop's own height survives.
    """
    image = np.full((1200, 1600, 3), 40, np.uint8)
    for start in range(0, 1600, 32):
        image[:, start:start + 8] = 220                   # lines a thumbnail would average away
    mask = np.zeros((1200, 1600), bool)
    mask[500:560, 700:820] = True
    segment, _info = cf.crop_to_mask()(Segment(image, mask=mask, occurrence_id="specimen0"))

    shown = gui(keys=[ord("1")])
    cf.exclusive_label_annotation(["good", "poor"], show_original=True)(segment)
    panel = shown.shown[-1][1]

    assert panel.shape[0] > np.asarray(segment.image).shape[0]
    original_width = int(image.shape[1] * panel.shape[0] / image.shape[0])
    row = panel[panel.shape[0] // 4, :original_width, 0]
    # at the crop's height no pixel is all line, so none stays this bright
    assert row.max() > 200 and row.min() < 60


def test_the_part_is_outlined_on_the_original():
    from critterframe.metrics.annotation import OUTLINE_COLOR, _original_view

    segment, image = a_cropped_segment()
    view = _original_view(segment, image.shape[0])

    assert view.shape == image.shape
    outlined = (view == OUTLINE_COLOR).all(axis=2)
    assert outlined.any()
    # the outline is on the part's edge, and nowhere near the far side of the photo
    assert not outlined[:, :150].any()
    assert (image == 40).any() and not (image == OUTLINE_COLOR).all(axis=2).any()   # the original is not drawn on


def test_an_untransformed_segment_has_nothing_to_add(gui):
    """With no transforms the working image is the original, so showing it twice says nothing."""
    plain = gui(keys=[ord("1")])
    cf.exclusive_label_annotation(["good", "poor"])(a_segment())
    with_original = gui(keys=[ord("1")])
    cf.exclusive_label_annotation(["good", "poor"], show_original=True)(a_segment())

    assert with_original.shown[-1][1].shape == plain.shown[-1][1].shape


def test_showing_the_original_is_not_part_of_the_recipe():
    """
    It changes what the annotator sees, not what a label is: turning it on must
    not make labels already given look like answers to a different question.
    """
    labels = ["good", "poor"]
    assert (cf.exclusive_label_annotation(labels).spec()
            == cf.exclusive_label_annotation(labels, show_original=True).spec())
