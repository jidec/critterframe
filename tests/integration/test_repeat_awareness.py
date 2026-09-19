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
        "attempted": SPECIMENS, "processed": SPECIMENS, "skipped": 0,
        "no_input": 0, "failed": 0, "failures": [], "flags": {},
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


def test_a_renamed_run_is_not_new_work(segmented_project):
    """
    name is recorded on the run but not part of identity (see Recipe.hash) --
    unlike a metric's run_name, nothing ever reads a segment mask BY run_name
    (masks.parquet's key is occurrence_id+part, full stop), so recognizing
    identical operations under a new name as the same work is a pure win: the
    run is recorded, but nothing is recomputed and the canonical mask stays
    exactly as it was.
    """
    result = cf.run_segments(segmented_project, run_name="second_pass",
                             steps=[cf.segment(ThresholdModel())],
                             visualize=False)["organism"]
    assert (result["processed"], result["skipped"]) == (0, SPECIMENS)


def test_a_default_named_canonical_and_reference_run_do_not_collide(segmented_project):
    """
    Both default to the same part ("organism") -- without folding reference
    into the default, they'd default to the identical name, and since
    resolve_recipe_currency is a no-op for segments nothing would catch a
    reference pass silently reading as though it superseded the canonical
    one in history.
    """
    canonical = run_records.load_runs(segmented_project, kind="segment").iloc[0]
    assert canonical["name"] == "organism"   # from the segmented_project template

    reference = cf.run_segments(segmented_project, from_part="organism",
                                steps=[cf.segment(ThresholdModel())],
                                reference=True, visualize=False)["organism"]
    assert reference["processed"] == SPECIMENS

    runs = run_records.load_runs(segmented_project, kind="segment")
    assert set(runs["name"]) == {"organism", "organism_reference"}

    # And the canonical recipe is still untouched and still recognized as done.
    rerun = segment_run(segmented_project)
    assert (rerun["processed"], rerun["skipped"]) == (0, SPECIMENS)


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


def test_a_renamed_metric_run_is_copied_and_its_own_column_is_populated(segmented_project):
    """
    Unlike a segment's run_name, a metric's run_name is the key every export
    column and records.metrics.latest_values read values back by -- so a
    rename must not silently do nothing (that would leave the new name's
    column empty forever). Instead the existing values are copied onto the
    new run_id: no model call, but the new name gets its own real rows.
    """
    metric_run(segmented_project)   # run_name="traits"
    result = cf.run_metrics(segmented_project, run_name="traits_v2",
                            metrics=[cf.body_length()], visualize=False)["organism"]
    assert (result["processed"], result["copied"]) == (0, SPECIMENS)

    original = cf.load_metrics(segmented_project, run_names=["traits"])
    renamed = cf.load_metrics(segmented_project, run_names=["traits_v2"])
    assert len(renamed) == SPECIMENS
    assert set(renamed["value"]) == set(original["value"])


# ---------------------------------------------------------------------------
# Group metrics: the reference population, not just the recipe, decides
# whether a stored score is still current.
# ---------------------------------------------------------------------------


def outlier_run(project_path, run_name="species_qc", **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(
        project_path, run_name=run_name,
        metrics=[cf.outlier(features=[cf.body_length()], from_run="traits")],
        **kwargs,
    )["organism"]


def test_a_group_metric_rerun_after_the_population_grows_rescopes_everything(
        segmented_project):
    """
    prepare() fits against context.occurrence_ids, which is deliberately not
    part of the hash -- so growing the reference population from 5 to 8 must
    invalidate the first 5's scores, fit against a population that no longer
    describes the project, rather than leaving them silently stale.
    """
    metric_run(segmented_project)   # 'traits': body_length for all 8

    first = outlier_run(segmented_project, limit=5)
    assert first["processed"] == 5

    second = outlier_run(segmented_project)
    assert second["processed"] == SPECIMENS
    assert second["copied"] == 0


def test_a_group_metric_run_under_a_new_name_with_the_same_population_is_copied(
        segmented_project):
    metric_run(segmented_project)
    first = outlier_run(segmented_project, run_name="species_qc")
    assert first["processed"] == SPECIMENS

    second = outlier_run(segmented_project, run_name="species_qc_v2")
    assert (second["processed"], second["copied"]) == (0, SPECIMENS)


def test_a_group_metric_run_under_a_new_name_with_a_different_population_recomputes(
        segmented_project):
    """
    A narrower population is a different fit, even at an unchanged recipe_hash
    -- copying the wider run's values here would attribute a score to a model
    this narrower run never actually fit.
    """
    metric_run(segmented_project)
    outlier_run(segmented_project, run_name="species_qc")

    result = outlier_run(segmented_project, run_name="species_qc_v2", limit=5)
    assert (result["processed"], result["copied"]) == (5, 0)


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


def _explodes(name="explodes"):
    """A metric that always raises, standing in for one that genuinely can."""
    from critterframe.recipes import Metric

    def _raise(_segment):
        raise ValueError("no")

    return Metric("explodes", _raise, version="1", unit="px", metric_name=name)


def test_a_metric_failure_is_not_retried_on_the_next_run(segmented_project):
    """
    The same guarantee segmentation already had. A metric that raised will
    raise again, and an embedding or annotation pass shouldn't spend the
    attempt rediscovering that -- so the failure is on record and the rerun
    skips it, saying how many it skipped.
    """
    from critterframe.records import failures as failure_records

    first = metric_run(segmented_project, metrics=[_explodes()])
    assert (first["failed"], first["previously_failed"]) == (SPECIMENS, 0)
    assert set(failure_records.load_failures(segmented_project)["stage"]) == {"metric"}

    second = metric_run(segmented_project, metrics=[_explodes()])
    assert (second["processed"], second["failed"]) == (0, 0)
    assert second["previously_failed"] == SPECIMENS


def test_retry_failed_attempts_them_again(segmented_project):
    """The escape hatch, for a failure whose cause was outside the recipe."""
    metric_run(segmented_project, metrics=[_explodes()])
    again = metric_run(segmented_project, metrics=[_explodes()], retry_failed=True)

    assert (again["failed"], again["previously_failed"]) == (SPECIMENS, 0)


def test_a_resegmentation_retries_a_failed_metric_by_itself(segmented_project):
    """
    The failure is keyed on the recipe AND the mask it was measured from, so
    replacing the mask is a different attempt -- no flag needed, the same way
    a changed URL retries a download.
    """
    metric_run(segmented_project, metrics=[_explodes()])
    segment_run(segmented_project, model=ThresholdModel(cutoff=90), force=True)

    after = metric_run(segmented_project, metrics=[_explodes()])
    assert (after["failed"], after["previously_failed"]) == (SPECIMENS, 0)


def test_a_succeeding_metric_clears_its_earlier_failure(segmented_project):
    """
    Hygiene, not correctness: the row can never match again, but leaving it
    there makes load_failures() a list of things that aren't failing.
    """
    from critterframe.records import failures as failure_records

    metric_run(segmented_project, metrics=[_explodes()])
    metric_run(segmented_project, metrics=[cf.body_length()], force=True)

    assert failure_records.load_failures(segmented_project, stage="metric").empty


@pytest.mark.slow
def test_a_reference_failure_is_not_erased_by_a_canonical_success(image_project):
    """
    records.failures keys on (occurrence_id, part, stage), so one stage for
    both passes would have a reference failure upsert over the canonical one,
    and a canonical success's clear_failures delete the reference's record.
    """
    from critterframe.records import failures as failure_records

    cf.run_segments(image_project, steps=[cf.segment(FailingModel())],
                    reference=True, visualize=False)
    reference_failures = failure_records.load_failures(image_project)
    assert set(reference_failures["stage"]) == {"segment_reference"}

    cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                    visualize=False)

    stages = failure_records.load_failures(image_project)["stage"]
    assert "segment_reference" in set(stages)
