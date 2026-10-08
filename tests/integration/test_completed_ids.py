"""
Which occurrences a run has finished.

The pool a review sample is drawn from is usually "whatever that run has got
through so far", and "finished" has to mean current: a mask that another recipe
has since replaced, or a value measured from a mask that no longer exists, is
not something this run can be said to have done.
"""

import pytest

import critterframe as cf
from critterframe.export import completed_ids
from critterframe.records import metrics as metric_records
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

IDS = [f"specimen{index}" for index in range(8)]


def test_a_segmentation_run_has_finished_what_it_masked(image_project):
    cf.define_subset(image_project, "first_five", occurrence_ids=IDS[:5])
    cf.run_segments(
        image_project,
        run_name="organisms",
        subset="first_five",
        steps=[cf.segment(ThresholdModel())],
        visualize=False,
    )

    assert completed_ids(image_project, "organisms") == IDS[:5]


def test_a_mask_another_run_replaced_is_no_longer_this_runs(segmented_project):
    """The first three are resegmented under another name; the original run keeps the other five."""
    cf.define_subset(segmented_project, "redo", occurrence_ids=IDS[:3])
    cf.run_segments(
        segmented_project,
        run_name="stricter",
        subset="redo",
        steps=[cf.segment(ThresholdModel(erode=2))],
        visualize=False,
    )

    assert completed_ids(segmented_project, "stricter") == IDS[:3]
    assert completed_ids(segmented_project, "organism") == IDS[3:]


def test_a_run_over_several_parts_has_to_be_told_which(segmented_project):
    cf.run_segments(
        segmented_project,
        run_name="parts",
        from_part="organism",
        outputs={"front": [cf.segment(ThresholdModel())], "back": [cf.segment(ThresholdModel(erode=1))]},
        visualize=False,
    )

    with pytest.raises(ValueError, match=r"covers parts \['back', 'front'\]"):
        completed_ids(segmented_project, "parts")
    assert completed_ids(segmented_project, "parts", part="front") == IDS
    with pytest.raises(KeyError, match="has no part 'wing'"):
        completed_ids(segmented_project, "parts", part="wing")


def test_a_metric_run_has_finished_what_it_has_a_current_value_for(segmented_project):
    cf.define_subset(segmented_project, "some", occurrence_ids=IDS[2:6])
    cf.run_metrics(
        segmented_project, run_name="lengths", subset="some", metrics=[cf.body_length()], visualize=False
    )
    assert completed_ids(segmented_project, "lengths") == IDS[2:6]

    # Two of those masks are replaced: their lengths describe masks the
    # project no longer holds, so the run has not finished them any more.
    cf.define_subset(segmented_project, "redo", occurrence_ids=IDS[2:4])
    cf.run_segments(
        segmented_project,
        run_name="stricter",
        subset="redo",
        steps=[cf.segment(ThresholdModel(erode=2))],
        visualize=False,
    )
    assert completed_ids(segmented_project, "lengths") == IDS[4:6]


def test_asking_who_is_done_reads_no_values(measured_project, monkeypatch):
    """
    An embedding run's values are gigabytes of JSON. Who has one is a question
    about rows, and answering it must not parse a single value.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("completed_ids parsed a stored value")

    monkeypatch.setattr(metric_records, "load_json", refuse)
    assert completed_ids(measured_project, "traits") == IDS
    assert "value" not in metric_records.result_keys(measured_project, "traits", "organism").columns


def test_an_unknown_run_says_what_runs_there_are(measured_project):
    with pytest.raises(KeyError, match="traits"):
        completed_ids(measured_project, "trait")


def test_a_name_shared_by_both_kinds_of_run_needs_the_kind(segmented_project):
    """Segmentation run names default to the part, so `organism` can easily be a metric run too."""
    cf.run_metrics(segmented_project, run_name="organism", metrics=[cf.body_length()], visualize=False)

    with pytest.raises(ValueError, match="both a segmentation and a metric run"):
        completed_ids(segmented_project, "organism")
    assert completed_ids(segmented_project, "organism", kind="segment") == IDS
    assert completed_ids(segmented_project, "organism", kind="metric") == IDS


def test_what_a_run_finished_is_a_pool_to_grow_a_sample_from(segmented_project):
    """
    The use it exists for: a review sample drawn from what a run has got
    through, with the subset's note saying what the pool was.
    """
    cf.define_subset(segmented_project, "some", occurrence_ids=IDS[:6])
    cf.run_metrics(
        segmented_project, run_name="lengths", subset="some", metrics=[cf.body_length()], visualize=False
    )

    sample = cf.grow_subset(
        segmented_project,
        "review",
        target_size=4,
        candidate_ids=cf.completed_ids(segmented_project, "lengths"),
        candidate_note="measured by lengths",
    )

    assert len(sample) == 4 and set(sample) <= set(IDS[:6])
    assert "measured by lengths (6 id(s))" in cf.load_subsets(segmented_project)["review"]["note"]
