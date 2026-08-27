"""
Deciding whether the pipeline is good enough, and on what evidence.

Three comparisons, all of which measure a METHOD rather than an organism, and
none of which persists anything:

  masks against reference masks (IoU) -- is the segmenter finding the right
  pixels;

  values against reference values -- do the traits measured from those masks
  agree with the ones a person measured by hand;

  an automated QC score against human labels -- can a threshold on it separate
  the crops that should be excluded from the ones that shouldn't, and at what
  cost in real data thrown away.

The word "reference" is load-bearing throughout: it is whatever you chose to
compare against, and calling it ground truth would assert the answer validation
exists to measure.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics import annotation
from critterframe.records import masks as mask_records
from critterframe.records.runs import load_runs
from helpers.models import ThresholdModel
from helpers.stubs import FakeCv2

pytestmark = pytest.mark.slow

SPECIMENS = 8


@pytest.fixture
def graded(measured_project):
    """
    A project with automated masks and traits, plus a stricter second
    segmenter's masks in the reference table -- the setup every comparison
    below reads.
    """
    cf.run_segments(measured_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, visualize=False)
    return measured_project


def test_masks_are_graded_against_the_reference_table(graded):
    scores = cf.validate_masks(graded, visualize=False)

    assert len(scores) == SPECIMENS
    assert scores["iou"].between(0, 1).all()
    assert scores["iou"].mean() < 1.0        # the two really do disagree


def test_the_reference_table_is_separate_from_what_the_project_processes(graded):
    """
    Both tables hold one mask per occurrence-part, and writing a reference one
    leaves the canonical one -- the mask everything downstream measures --
    exactly as it was.
    """
    canonical = mask_records.load_masks(graded)
    reference = mask_records.load_masks(graded, reference=True)

    assert len(canonical) == len(reference) == SPECIMENS
    assert set(canonical["recipe_hash"]).isdisjoint(set(reference["recipe_hash"]))


def test_traits_can_be_measured_from_the_reference_masks_too(graded):
    """
    Which is what makes the second comparison possible: the same metric recipe
    over the other table, and the two runs stay distinguishable because the
    mask table is part of the recipe's identity.
    """
    result = cf.run_metrics(graded, run_name="reference_traits",
                            transforms=[cf.remove_appendages(), cf.orient()],
                            metrics=[cf.body_length()],
                            reference=True, visualize=False)["organism"]
    assert result["processed"] == SPECIMENS

    exported = cf.export_metrics(graded)
    assert "traits__organism__body_length" in exported.columns
    assert "reference_traits__organism__body_length" in exported.columns


def test_measuring_both_tables_is_two_runs_not_one(graded):
    """
    `inputs={"masks": ...}` is in the recipe hash, so the second run is not
    mistaken for a repeat of the first -- which it would be, since every other
    part of the recipe is identical.
    """
    cf.run_metrics(graded, run_name="both", metrics=[cf.body_length()],
                   visualize=False)
    second = cf.run_metrics(graded, run_name="both", metrics=[cf.body_length()],
                            reference=True, visualize=False)["organism"]

    assert second["processed"] == SPECIMENS
    assert load_runs(graded, name="both")["recipe_hash"].nunique() == 2


def test_predicted_and_reference_values_are_compared_per_metric(graded):
    cf.run_metrics(graded, run_name="reference_traits",
                   transforms=[cf.remove_appendages(), cf.orient()],
                   metrics=[cf.mask_area(name="area_px", unit="px2")],
                   reference=True, visualize=False)

    comparison = cf.compare_metrics(graded, "traits", "reference_traits",
                                    metric_names=["area_px"])
    row = comparison.iloc[0]

    # Area rather than length, because an eroded mask is unambiguously smaller
    # while "length" on a near-symmetric drawn ellipse depends on which axis
    # orient() picked (see test_pipeline_core).
    assert row["n"] == SPECIMENS
    assert row["mean_abs_diff"] > 0          # a tighter mask is a smaller one
    assert row["bias"] > 0                   # and consistently so
    assert -1 <= row["correlation"] <= 1


def test_a_human_measurement_grades_the_pipeline_across_names(graded, monkeypatch):
    """
    The mapping form of `metric_names`: a clicked distance and a measured body
    length are the same quantity recorded two ways, and both keep their own
    column rather than colliding under one name.
    """
    # Three of each per occurrence: the collecting loop polls once per click,
    # and the cosmetic wait that holds the second marker on screen consumes one
    # more -- which the callback ignores, having its two points already.
    clicks = [(cv2.EVENT_LBUTTONDOWN, 120, 60),
              (cv2.EVENT_LBUTTONDOWN, 120, 160),
              (cv2.EVENT_MOUSEMOVE, 0, 0)] * SPECIMENS
    monkeypatch.setattr(annotation, "cv2",
                        FakeCv2(keys=[ord(" ")] * (SPECIMENS * 3),
                                clicks=clicks))

    cf.run_metrics(graded, run_name="by_hand",
                   metrics=[cf.click_two_points()], visualize=False)

    comparison = cf.compare_metrics(
        graded, "traits", "by_hand",
        metric_names={"body_length": "click_two_points__length_px"})

    assert comparison.iloc[0]["reference_metric"] == "click_two_points__length_px"
    assert comparison.iloc[0]["n"] == SPECIMENS


def test_a_qc_threshold_is_calibrated_against_human_labels(graded, monkeypatch):
    """
    The third comparison, and the one that decides what an export throws away.
    A threshold is picked by looking at what it WOULD have excluded on crops a
    person already judged -- never by eye, and never hardcoded.
    """
    from critterframe.export import metrics_wide
    from critterframe.validation.filters import suggest_threshold, sweep_thresholds

    # Two of the eight are called unusable; the rest are fine.
    flags = [ord("1")] * SPECIMENS
    flags[0] = ord("3")            # cut_off
    flags[1] = ord("2")            # not_an_organism
    monkeypatch.setattr(annotation, "cv2", FakeCv2(keys=flags))

    cf.run_metrics(graded, run_name="screening",
                   metrics=[cf.annotate_flags()], visualize=False)

    wide = metrics_wide(graded)
    sweep = sweep_thresholds(wide, "traits__organism__blur_variance", "below",
                             "screening__organism__annotate_flags")

    assert not sweep.empty
    assert set(sweep["n_bad"]) == {2}
    # A suggestion is a row to read, and None is a legitimate answer meaning
    # this metric can't separate the two groups well enough for what you asked.
    chosen = suggest_threshold(sweep, max_fpr=0.0)
    assert chosen is None or chosen["fpr"] == 0.0


def test_the_labels_pick_out_the_crops_worth_more_human_time(graded, monkeypatch):
    """
    Screening comes first because it says which crops the expensive human
    questions can even be asked of: a crop with no single complete organism has
    no boundary to correct and no body axis to click.
    """
    flags = [ord("1")] * SPECIMENS
    flags[0] = ord("2")
    monkeypatch.setattr(annotation, "cv2", FakeCv2(keys=flags))

    cf.run_metrics(graded, run_name="screening",
                   metrics=[cf.annotate_flags()], visualize=False)

    usable = cf.occurrences_matching(graded, "screening",
                                     {"annotate_flags": "usable"})
    assert len(usable) == SPECIMENS - 1

    cf.define_subset(graded, "usable", occurrence_ids=usable)
    assert len(cf.export_metrics(graded, subset="usable")) == SPECIMENS - 1


def test_validation_leaves_no_trace_in_the_project(graded):
    """
    All comparison, nothing persisted. A validation score is a judgement about
    a method, and storing it as a metric would put it in the trait table beside
    the organisms' own measurements.
    """
    runs_before = len(load_runs(graded))
    values_before = len(cf.export_metrics(graded).columns)

    cf.validate_masks(graded, visualize=False)
    cf.compare_metrics(graded, "traits", "traits")

    assert len(load_runs(graded)) == runs_before
    assert len(cf.export_metrics(graded).columns) == values_before


def test_a_filter_calibrated_this_way_narrows_the_export_and_deletes_nothing(
        graded):
    """
    The end of the road: a threshold becomes an export filter, the export gets
    smaller, and the project still holds every occurrence -- so tomorrow's
    threshold can be different.
    """
    column = "traits__organism__blur_variance"
    everything = cf.export_metrics(graded)
    cutoff = float(np.median(everything[column]))

    filtered = cf.export_metrics(graded, filters={column: (">", cutoff)})
    assert 0 < len(filtered) < SPECIMENS
    assert len(cf.export_metrics(graded)) == SPECIMENS
