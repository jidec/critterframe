"""
Painting a mask by hand -- an alternative segmentation, not a separate system.

`draw_mask()` and `correct_mask()` are `Segmentation` operations returning
(segment, info) and feeding the same mask table as any model. What differs is
only which one a run's recipe names, and therefore which recipe hash the
resulting mask carries. Everything in this file is about that equivalence, and
about the numbers the correction produces -- which are a grade of the segmenter
being corrected, and the reason there is no separate metric asking a person to
rate a mask.

The window is stubbed the same way as in `test_annotation.py`: on the module
under test, with `waitKey` failing loudly rather than blocking, because the real
loop has no timeout.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.segmentation import manual
from helpers.stubs import FakeCv2

SAVE = ord("s")
CANCEL = 27


def a_segment(with_mask=True, panel_sink=None):
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True                # a 40x40 block, 1600 px
    image = np.zeros((100, 100, 3), np.uint8)
    image[mask] = 200
    return Segment(image, mask=mask if with_mask else None,
                   occurrence_id="specimen0", panel_sink=panel_sink)


@pytest.fixture
def gui(monkeypatch):
    def install(keys=(), clicks=()):
        fake = FakeCv2(keys=keys, clicks=clicks)
        monkeypatch.setattr(manual, "cv2", fake)
        return fake
    return install


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_both_are_segmentations_like_any_model():
    assert cf.draw_mask().kind == "segment"
    assert cf.correct_mask().kind == "segment"


def test_starting_empty_is_what_separates_them():
    """
    A silent difference worth knowing about, since the window looks the same
    either way -- so it is in the parameters, and therefore in the hash.
    """
    assert cf.draw_mask().parameters["start_empty"] is True
    assert cf.correct_mask().parameters["start_empty"] is False
    assert cf.draw_mask().spec() != cf.correct_mask().spec()


def test_the_brush_size_is_part_of_the_recipe():
    assert cf.draw_mask(brush_radius=4).spec() != cf.draw_mask(brush_radius=12).spec()


def test_the_factories_open_no_window():
    """Configuring a recipe must not block on a person."""
    cf.draw_mask()
    cf.correct_mask()


# ---------------------------------------------------------------------------
# Saving and cancelling
# ---------------------------------------------------------------------------


def test_saving_an_untouched_mask_changes_nothing(gui):
    gui(keys=[SAVE])
    corrected, info = cf.correct_mask()(a_segment())

    assert np.array_equal(corrected.mask, a_segment().mask)
    assert info["removed_fraction"] == 0.0
    assert info["added_fraction"] == 0.0
    assert info["iou"] == 1.0
    assert info["cancelled"] is False


def test_escape_keeps_the_mask_it_started_with(gui):
    """
    Cancel means "leave this one alone", which has to be distinguishable from
    "I looked and it was already right" -- hence the flag.
    """
    gui(keys=[CANCEL], clicks=[(cv2.EVENT_LBUTTONDOWN, 50, 50)])
    corrected, info = cf.correct_mask()(a_segment())

    assert np.array_equal(corrected.mask, a_segment().mask)
    assert info["cancelled"] is True


def test_erasing_removes_pixels_and_reports_how_many(gui):
    """
    Left-drag erases what the segmenter wrongly included -- an extra leg, a
    shadow, a second organism in frame.
    """
    gui(keys=[SAVE], clicks=[(cv2.EVENT_LBUTTONDOWN, 40, 40)])
    corrected, info = cf.correct_mask(brush_radius=8)(a_segment())

    assert corrected.mask.sum() < 1600
    assert 0 < info["removed_fraction"] < 1
    assert info["added_fraction"] == 0.0
    assert info["iou"] < 1.0


def test_painting_adds_pixels_and_reports_them_separately(gui):
    """
    Both of a segmenter's failure directions, kept apart: a mask that was 20%
    too big and one that was 20% too small are not the same failure, and one
    net figure would call them the same.
    """
    gui(keys=[SAVE], clicks=[(cv2.EVENT_RBUTTONDOWN, 10, 10)])
    corrected, info = cf.correct_mask(brush_radius=6)(a_segment())

    assert corrected.mask.sum() > 1600
    assert info["added_fraction"] > 0
    assert info["removed_fraction"] == 0.0


def test_a_drag_paints_along_its_path(gui):
    """
    Mouse-move while a button is down keeps painting -- otherwise correcting a
    boundary would be a click per pixel.
    """
    gui(keys=[SAVE],
        clicks=[(cv2.EVENT_RBUTTONDOWN, 10, 10),
                (cv2.EVENT_MOUSEMOVE, 15, 10),
                (cv2.EVENT_MOUSEMOVE, 20, 10)])
    _corrected, info = cf.correct_mask(brush_radius=3)(a_segment())
    assert info["added_fraction"] > 0


def test_releasing_the_button_stops_the_painting(gui):
    """A move with no button down must not draw."""
    gui(keys=[SAVE],
        clicks=[(cv2.EVENT_RBUTTONDOWN, 10, 10),
                (cv2.EVENT_RBUTTONUP, 10, 10),
                (cv2.EVENT_MOUSEMOVE, 80, 80)])
    corrected, _info = cf.correct_mask(brush_radius=3)(a_segment())
    assert not corrected.mask[75:85, 75:85].any()


def test_drawing_from_scratch_starts_from_nothing(gui):
    """
    Even when the segment arrives with a mask -- which is exactly the case
    where the difference between the two operations matters.
    """
    gui(keys=[SAVE], clicks=[(cv2.EVENT_RBUTTONDOWN, 10, 10)])
    drawn, _info = cf.draw_mask(brush_radius=6)(a_segment())

    assert drawn.mask.sum() < 200            # just the brush, not the block
    assert not drawn.mask[50, 50]


def test_saving_nothing_at_all_is_refused(gui):
    """
    An empty mask is not a segmentation, and storing one would put a
    zero-area organism in the reference table.
    """
    gui(keys=[SAVE])
    with pytest.raises(ValueError, match="no mask was drawn"):
        cf.draw_mask()(a_segment())


def test_plus_and_minus_resize_the_brush_mid_session(gui):
    """
    A single right-click at a fixed point paints a bigger blob once the brush
    has been grown, which is the only externally visible sign the resize took.

    FakeCv2 pops one scripted click per waitKey() call regardless of which key
    that call returns, so the click has to be padded with no-op moves to land
    on the waitKey() call that actually follows both '+' presses.
    """
    noop = (cv2.EVENT_MOUSEMOVE, 0, 0)
    gui(keys=[ord("+"), ord("+"), SAVE],
        clicks=[noop, noop, (cv2.EVENT_RBUTTONDOWN, 50, 50)])
    grown, _info = cf.correct_mask(brush_radius=2)(a_segment(with_mask=False))

    gui(keys=[SAVE], clicks=[(cv2.EVENT_RBUTTONDOWN, 50, 50)])
    plain, _info = cf.correct_mask(brush_radius=2)(a_segment(with_mask=False))

    assert grown.mask.sum() > plain.mask.sum()


def test_minus_shrinks_and_will_not_go_below_one(gui):
    noop = (cv2.EVENT_MOUSEMOVE, 0, 0)
    gui(keys=[ord("-")] * 5 + [SAVE],
        clicks=[noop] * 5 + [(cv2.EVENT_RBUTTONDOWN, 50, 50)])
    corrected, _info = cf.correct_mask(brush_radius=2)(a_segment(with_mask=False))

    # radius floors at 1 rather than 1 - 5 = -4, so some paint still lands.
    assert corrected.mask.sum() > 0


def test_resizing_does_not_change_the_recipe_hash(gui):
    """
    The starting brush_radius is in spec(); an in-session resize is an
    operator action during an already-nondeterministic operation, not a
    parameter, so it must not move the hash.
    """
    before = cf.correct_mask(brush_radius=5).spec()
    gui(keys=[ord("+"), ord("-"), SAVE])
    cf.correct_mask(brush_radius=5)(a_segment())
    after = cf.correct_mask(brush_radius=5).spec()
    assert before == after


def test_a_key_that_means_nothing_is_ignored(gui):
    gui(keys=[ord("x"), SAVE])
    _corrected, info = cf.correct_mask()(a_segment())
    assert info["cancelled"] is False


def test_the_window_is_closed_afterwards(gui):
    fake = gui(keys=[SAVE])
    cf.correct_mask()(a_segment())
    assert fake.destroyed


def test_a_stub_that_runs_dry_fails_instead_of_hanging(gui):
    gui(keys=[])
    with pytest.raises(AssertionError, match="ran dry"):
        cf.correct_mask()(a_segment())


# ---------------------------------------------------------------------------
# What it leaves behind
# ---------------------------------------------------------------------------


def test_the_correction_is_the_grade(gui):
    """
    Which is why there is no metric asking a person to rate a mask: paint the
    boundary, and the IoU against what the segmenter produced falls out of the
    correction itself.
    """
    gui(keys=[SAVE], clicks=[(cv2.EVENT_LBUTTONDOWN, 40, 40)])
    _corrected, info = cf.correct_mask(brush_radius=10)(a_segment())

    assert 0 < info["iou"] < 1
    assert info["area_after"] < info["area_before"]


def test_a_panel_shows_what_was_kept_erased_and_added(gui):
    class RecordingSink:
        def __init__(self):
            self.stages = []

        def collect(self, occurrence_id, stage, image):
            self.stages.append(stage)
            assert image.dtype == np.uint8

    sink = RecordingSink()
    gui(keys=[SAVE], clicks=[(cv2.EVENT_LBUTTONDOWN, 40, 40)])
    cf.correct_mask()(a_segment(panel_sink=sink))
    assert sink.stages


@pytest.mark.slow
def test_a_hand_drawn_mask_lands_in_the_reference_table(gui, segmented_project):
    """
    The workflow the operation exists for: correct a few dozen by hand into the
    reference table, then validate the automated masks against them. Same run
    machinery, same repeat-awareness, different table.

    force= is explicit because a hand-drawn mask is not reproducible; see
    test_a_hand_drawn_recipe_will_not_guess_about_completed_work below.
    """
    from critterframe.records import masks as mask_records

    gui(keys=[SAVE] * 3)
    result = cf.run_segments(segmented_project, run_name="by_hand",
                             steps=[cf.correct_mask()], from_part="organism",
                             reference=True, limit=3, visualize=False,
                             force=False)["organism"]

    assert result["processed"] == 3
    assert len(mask_records.load_masks(segmented_project, reference=True)) == 3
    assert len(mask_records.load_masks(segmented_project)) == 8   # untouched


@pytest.mark.slow
def test_a_hand_drawn_recipe_will_not_guess_about_completed_work(gui,
                                                                 segmented_project):
    """
    Two people painting one crop produce two different masks under one recipe
    hash, so "already done by this recipe" cannot be read either way. The run
    refuses rather than picking one, and both answers stay reachable.
    """
    from critterframe.records import masks as mask_records

    drawn = dict(run_name="by_hand", steps=[cf.correct_mask()],
                 from_part="organism", reference=True, limit=3, visualize=False)

    gui(keys=[SAVE] * 3)
    with pytest.raises(ValueError, match="correct_mask is not deterministic"):
        cf.run_segments(segmented_project, **drawn)

    gui(keys=[SAVE] * 3)
    first = cf.run_segments(segmented_project, force=False, **drawn)["organism"]
    assert first["processed"] == 3

    # force=False resumes: the three already painted are not painted again.
    gui(keys=[])
    resumed = cf.run_segments(segmented_project, force=False, **drawn)["organism"]
    assert (resumed["processed"], resumed["skipped"]) == (0, 3)

    # force=True is the second pass, over the same occurrence-parts.
    gui(keys=[SAVE] * 3)
    redone = cf.run_segments(segmented_project, force=True, **drawn)["organism"]
    assert redone["processed"] == 3
    assert len(mask_records.load_masks(segmented_project, reference=True)) == 3


@pytest.mark.slow
def test_a_model_recipe_still_defaults_to_resuming(segmented_project):
    """
    The refusal is scoped to the operations that earn it: a recipe of
    reproducible operations skips completed work with no force= at all, exactly
    as it did before draw_mask gained the flag.
    """
    from helpers.models import ThresholdModel

    steps = [cf.segment(ThresholdModel())]
    cf.run_segments(segmented_project, run_name="again", steps=steps,
                    visualize=False)
    second = cf.run_segments(segmented_project, run_name="again", steps=steps,
                             visualize=False)["organism"]
    assert (second["processed"], second["skipped"]) == (0, 8)
