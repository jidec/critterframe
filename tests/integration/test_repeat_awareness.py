"""
Work already done is not done twice.

`run_segments` and `run_metrics` both compute their recipe hash, ask the store
which (occurrence_id, part) pairs that hash already covers, and skip them. That
is what makes runs interruptible and makes an expensive metric behave like
cached derived data -- and it is a behavioural guarantee, not an optimization,
so it is asserted rather than eyeballed.

These assertions were already written, in English, in the smoke script:
`scripts/simple_tests/pipeline_synthetic_test.py` prints "<- expect processed=0,
skipped=8 (repeat-aware)" and trusts a human to notice when it says otherwise.

The guarantee rests on an assumption the hash cannot check: that running a
recipe twice would produce the same thing. Where an operation says it would not,
the last test here shows the run refusing to apply the shortcut on its own.
"""

import pytest

import critterframe as cf
from critterframe.records import runs as run_records
from helpers.models import FailingModel, ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8


def segment_run(project_path, model=None, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_segments(project_path,
                           steps=[cf.segment(model or ThresholdModel())],
                           **kwargs)["organism"]


def metric_run(project_path, metrics=None, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(project_path, run_name="traits",
                          metrics=metrics or [cf.body_length()],
                          **kwargs)["organism"]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_the_first_run_processes_everything(image_project):
    assert segment_run(image_project) == {
        "processed": SPECIMENS, "skipped": 0, "failed": 0,
        "previously_failed": 0, "run_id": 1}


def test_an_identical_rerun_does_no_work(segmented_project):
    """The guarantee. Same recipe, same occurrences, nothing recomputed."""
    result = segment_run(segmented_project)
    assert (result["processed"], result["skipped"]) == (0, SPECIMENS)


def test_force_overrides_the_skip(segmented_project):
    result = segment_run(segmented_project, force=True)
    assert (result["processed"], result["skipped"]) == (SPECIMENS, 0)


def test_a_changed_model_parameter_is_new_work(segmented_project):
    """
    The model's identity() is in the recipe hash, so a different configuration
    is a different recipe and none of the stored masks answer for it.
    """
    result = segment_run(segmented_project, model=ThresholdModel(erode=2))
    assert (result["processed"], result["skipped"]) == (SPECIMENS, 0)


def test_a_changed_run_name_is_new_work(segmented_project):
    """
    Deliberately rerunning the same operations under a new name records a
    genuinely new run rather than being skipped -- the name is part of identity
    because it is also what the export column is called.
    """
    result = cf.run_segments(segmented_project, run_name="second_pass",
                             steps=[cf.segment(ThresholdModel())],
                             visualize=False)["organism"]
    assert result["processed"] == SPECIMENS


def test_an_interrupted_run_resumes_where_it_stopped(image_project):
    """
    Why completion is checked per occurrence rather than per run: a run that
    covered three specimens and died leaves three fewer to do, not a project
    that has to start over.
    """
    partial = segment_run(image_project, limit=3)
    assert partial["processed"] == 3

    resumed = segment_run(image_project)
    assert (resumed["processed"], resumed["skipped"]) == (SPECIMENS - 3, 3)


def test_a_subset_run_and_a_whole_project_run_are_the_same_work(image_project):
    """
    The subset is recorded on the run but deliberately not hashed: processing
    the rest of the project later continues the same work.
    """
    cf.define_subset(image_project, "boxA", column="device", values=["boxA"])
    first = segment_run(image_project, subset="boxA")
    second = segment_run(image_project)

    assert first["processed"] == 4
    assert (second["processed"], second["skipped"]) == (4, 4)


def test_every_failure_is_counted_and_none_is_fatal(image_project):
    """
    One bad occurrence must not cost a run. Eight bad ones must not either --
    they come back as a count, not an exception.
    """
    result = segment_run(image_project, model=FailingModel())
    assert (result["processed"], result["failed"]) == (0, SPECIMENS)


def test_a_failed_segmentation_is_not_retried_until_asked(image_project):
    """
    A failure is recorded too, not just logged -- so a rerun against the same
    recipe does not send the model the same occurrence a second time.
    """
    model = FailingModel()
    first = segment_run(image_project, model=model)
    assert first["failed"] == SPECIMENS
    calls_after_first = model.calls

    second = segment_run(image_project, model=model)
    assert (second["processed"], second["failed"]) == (0, 0)
    assert (second["skipped"], second["previously_failed"]) == (SPECIMENS, SPECIMENS)
    assert model.calls == calls_after_first

    retried = segment_run(image_project, model=model, retry_failed=True)
    assert retried["failed"] == SPECIMENS
    assert model.calls == calls_after_first * 2


def test_a_changed_recipe_retries_a_failure_without_being_asked(image_project):
    """
    The failure is scoped to the recipe that produced it, the same way a mask
    is -- a retuned recipe is automatically new work, no retry_failed needed.
    """
    model = FailingModel()
    cf.run_segments(image_project, steps=[cf.segment(model, mask_threshold=0.0)],
                    visualize=False)
    calls_after_first = model.calls

    result = cf.run_segments(image_project,
                             steps=[cf.segment(model, mask_threshold=0.1)],
                             visualize=False)["organism"]
    assert result["failed"] == SPECIMENS
    assert model.calls == calls_after_first * 2


def test_a_skipped_rerun_still_records_a_run(segmented_project):
    """
    A run that did nothing is still a run that happened, and the record is what
    says "this recipe was asked for again and had nothing left to do".
    """
    segment_run(segmented_project)
    runs = run_records.load_runs(segmented_project, kind="segment")
    assert len(runs) == 2
    assert runs.iloc[0]["n_skipped"] == SPECIMENS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_an_identical_metric_rerun_does_no_work(segmented_project):
    first = metric_run(segmented_project)
    second = metric_run(segmented_project)

    assert (first["processed"], first["skipped"]) == (SPECIMENS, 0)
    assert (second["processed"], second["skipped"]) == (0, SPECIMENS)


def test_adding_a_metric_to_a_recipe_is_new_work(segmented_project):
    """
    The operation list is in the hash, so a recipe measuring two traits is not
    the recipe that measured one -- and the second trait would otherwise never
    be computed for the occurrences the first one covered. force=True because
    this deliberately moves "traits" onto a different recipe -- see
    records.runs.resolve_recipe_currency.
    """
    metric_run(segmented_project, metrics=[cf.body_length()])
    result = metric_run(segmented_project,
                        metrics=[cf.body_length(), cf.max_width()], force=True)
    assert result["processed"] == SPECIMENS


def test_a_transform_before_the_metrics_is_part_of_the_recipe(segmented_project):
    """
    Measuring an oriented segment is different work from measuring a raw one,
    even though the metric operation is the same. force=True acknowledges
    moving "traits" onto that different recipe.
    """
    metric_run(segmented_project)
    result = metric_run(segmented_project, transforms=[cf.orient()], force=True)
    assert result["processed"] == SPECIMENS


def test_occurrences_without_a_mask_are_neither_measured_nor_counted_done(
        image_project, caplog):
    """
    Nothing to measure is not a failure and not a skip -- it is an occurrence
    segmentation hasn't reached, and the run says so out loud.
    """
    segment_run(image_project, limit=3)
    with caplog.at_level("WARNING"):
        result = metric_run(image_project)

    assert (result["processed"], result["skipped"], result["failed"]) == (3, 0, 0)
    assert "have no 'organism' mask" in caplog.text


# ---------------------------------------------------------------------------
# Where the assumption underneath does not hold
# ---------------------------------------------------------------------------


def test_a_recipe_that_would_not_reproduce_itself_refuses_the_shortcut(
        segmented_project):
    """
    Skipping is sound only because an identical hash means identical work. A
    hand-drawn mask breaks that: two people painting one crop hash alike and
    produce different masks, so "already covered by this recipe" is genuinely
    ambiguous -- resume, or a second pass? The run will not pick for you, and
    both readings stay reachable.
    """
    with pytest.raises(ValueError, match="not deterministic"):
        cf.run_segments(segmented_project, run_name="by_hand",
                        steps=[cf.draw_mask()], visualize=False)

    # Nothing else changes: a recipe of reproducible operations still skips
    # completed work without being asked twice.
    segment_run(segmented_project)
    assert segment_run(segmented_project)["skipped"] == SPECIMENS
