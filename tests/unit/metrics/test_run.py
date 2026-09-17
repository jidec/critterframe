"""
`_completed_keys`'s name scope, `_copyable_rows`'s newest-run grouping, and
`_current_for_population`'s group-metric staleness check -- the three pieces
behind run_metrics()'s rename handling (see tests/integration/test_repeat_awareness.py
for the end-to-end behavior these compose into) -- plus run_metrics()'s
run_name default, a separate, signature-level convenience.

Rows for the first three are written directly through
records.runs/records.metrics rather than a full run_metrics() call, so each
piece is exercised in isolation from image loading, masks, and the
per-occurrence loop. The run_name default tests need the real thing, since
what they're checking is what run_metrics() itself resolves the name to.
"""

import pytest

import critterframe as cf
from critterframe.metrics import run as metrics_run
from critterframe.metrics.dimensions import body_length, max_width
from critterframe.records import metrics as metric_records
from critterframe.records import runs as run_records
from critterframe.recipes import Metric, Recipe
from helpers.models import ThresholdModel


def a_recipe(name="traits", part="organism"):
    return Recipe("metric", name, [body_length()], part=part)


def write(project_path, run_name, rows, context=None, part="organism"):
    """Record one run's worth of metric rows directly, returning (run_id, recipe_hash)."""
    recipe = a_recipe(name=run_name, part=part)
    run_id = run_records.start_run(project_path, recipe, context=context)
    metric_records.append_metrics(project_path, run_id, recipe.hash, rows)
    run_records.finish_run(project_path, run_id, processed=len(rows))
    return run_id, recipe.hash


# ---------------------------------------------------------------------------
# _completed_keys -- scoped by name, unlike records.masks.completed_keys
# ---------------------------------------------------------------------------


def test_completed_keys_with_no_name_asks_the_weaker_question(tmp_path):
    rows = [metric_records.make_metric_row("a", "organism", "body_length", 1.0)]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)

    assert metrics_run._completed_keys(tmp_path, recipe_hash) == {("a", "organism")}


def test_completed_keys_scoped_to_a_name_ignores_another_names_work(tmp_path):
    """
    Since name isn't part of the recipe hash, an unscoped check would see
    'beta' as already covered by 'alpha''s rows -- exactly the silent-empty-
    column failure the name scope exists to prevent.
    """
    rows = [metric_records.make_metric_row("a", "organism", "body_length", 1.0)]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)

    assert metrics_run._completed_keys(tmp_path, recipe_hash, run_name="alpha") == \
        {("a", "organism")}
    assert metrics_run._completed_keys(tmp_path, recipe_hash, run_name="beta") == set()


# ---------------------------------------------------------------------------
# _copyable_rows -- what a rename copies instead of recomputing
# ---------------------------------------------------------------------------


def test_copyable_rows_reads_an_existing_names_values(tmp_path):
    rows = [metric_records.make_metric_row("a", "organism", "body_length", 1.0)]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)

    copyable = metrics_run._copyable_rows(tmp_path, recipe_hash, {("a", "organism")},
                                          "organism")
    assert copyable["a"][0]["metric_name"] == "body_length"
    assert copyable["a"][0]["value"] == 1.0


def test_copyable_rows_keeps_the_whole_batch_of_the_newest_run(tmp_path):
    """
    An occurrence's values under one recipe_hash are always written together
    in one run -- so when two runs share a hash, copying must take EVERY
    metric_name from whichever run is newest, not just the first row seen for
    that occurrence (a naive "already seen this occurrence" check would drop
    the rest of that same run's own rows).
    """
    older = [metric_records.make_metric_row("a", "organism", "body_length", 1.0),
             metric_records.make_metric_row("a", "organism", "max_width", 2.0)]
    write(tmp_path, "alpha", older)

    newer = [metric_records.make_metric_row("a", "organism", "body_length", 9.0),
             metric_records.make_metric_row("a", "organism", "max_width", 8.0)]
    _run_id, recipe_hash = write(tmp_path, "beta", newer)

    copyable = metrics_run._copyable_rows(tmp_path, recipe_hash, {("a", "organism")},
                                          "organism")
    values = {row["metric_name"]: row["value"] for row in copyable["a"]}
    assert values == {"body_length": 9.0, "max_width": 8.0}


def test_copyable_rows_only_returns_the_requested_keys(tmp_path):
    rows = [metric_records.make_metric_row("a", "organism", "body_length", 1.0),
            metric_records.make_metric_row("b", "organism", "body_length", 2.0)]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)

    copyable = metrics_run._copyable_rows(tmp_path, recipe_hash, {("a", "organism")},
                                          "organism")
    assert list(copyable) == ["a"]


def test_copyable_rows_of_nothing_is_empty(tmp_path):
    assert metrics_run._copyable_rows(tmp_path, "whatever", set(), "organism") == {}


# ---------------------------------------------------------------------------
# _current_for_population -- group-metric staleness, shared by both scopes
# ---------------------------------------------------------------------------


def test_an_empty_prepared_is_a_no_op(tmp_path):
    """
    An ordinary metric's value never depends on who else was in scope, so a
    recipe with no group metric (prepared={}) skips this check entirely.
    """
    rows = [metric_records.make_metric_row("a", "organism", "body_length", 1.0)]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)

    keys = {("a", "organism")}
    assert metrics_run._current_for_population(
        tmp_path, keys, recipe_hash, "organism", {}) == keys


def test_a_matching_population_passes(tmp_path):
    rows = [metric_records.make_metric_row("a", "organism", "outlier",
                                           {"is_outlier": False})]
    _run_id, recipe_hash = write(
        tmp_path, "alpha", rows,
        context={"operations": {"outlier": {"population": {"ids_hash": "pop1"}}}})

    prepared = {"outlier": {"population": {"ids_hash": "pop1"}}}
    assert metrics_run._current_for_population(
        tmp_path, {("a", "organism")}, recipe_hash, "organism", prepared) == \
        {("a", "organism")}


def test_a_different_population_fails(tmp_path):
    """
    The recipe hash matches -- name and everything else about the recipe is
    identical -- but the reference population the source run fit against was
    not, so the stored score isn't safe to treat as current or to copy.
    """
    rows = [metric_records.make_metric_row("a", "organism", "outlier",
                                           {"is_outlier": False})]
    _run_id, recipe_hash = write(
        tmp_path, "alpha", rows,
        context={"operations": {"outlier": {"population": {"ids_hash": "pop1"}}}})

    prepared = {"outlier": {"population": {"ids_hash": "pop2"}}}
    assert metrics_run._current_for_population(
        tmp_path, {("a", "organism")}, recipe_hash, "organism", prepared) == set()


def test_a_source_run_with_no_recorded_population_fails_conservatively(tmp_path):
    """
    A run written before this tracking existed, or with no group metric of its
    own, has nothing to confirm the population against -- treated the same as
    a mismatch, never silently trusted.
    """
    rows = [metric_records.make_metric_row("a", "organism", "outlier",
                                           {"is_outlier": False})]
    _run_id, recipe_hash = write(tmp_path, "alpha", rows)  # no context at all

    prepared = {"outlier": {"population": {"ids_hash": "pop1"}}}
    assert metrics_run._current_for_population(
        tmp_path, {("a", "organism")}, recipe_hash, "organism", prepared) == set()


# ---------------------------------------------------------------------------
# run_metrics' run_name default -- a signature-level convenience, distinct
# from the rename/copy machinery above but living in the same module.
# ---------------------------------------------------------------------------


def test_a_single_metric_defaults_its_run_name(segmented_project):
    cf.run_metrics(segmented_project, metrics=[cf.body_length()], visualize=False)
    runs = run_records.load_runs(segmented_project, kind="metric")
    assert runs.iloc[0]["name"] == "body_length"


def test_the_default_respects_a_custom_metric_name(segmented_project):
    """The default follows metric_name, not the operation's own name, so a
    caller who already renamed the metric doesn't have the run renamed back."""
    cf.run_metrics(segmented_project, metrics=[cf.mask_area(name="area_px")],
                   visualize=False)
    runs = run_records.load_runs(segmented_project, kind="metric")
    assert runs.iloc[0]["name"] == "area_px"


def test_more_than_one_metric_needs_an_explicit_run_name(segmented_project):
    """
    No single obvious name for a combined result -- guessing one (joining
    names, say) would silently rename an export column the moment a second
    metric gets added to an existing call, so this asks instead of guessing.
    """
    with pytest.raises(ValueError, match="needs an explicit run_name"):
        cf.run_metrics(segmented_project, metrics=[body_length(), max_width()],
                       visualize=False)


def test_an_explicit_run_name_still_overrides_the_default(segmented_project):
    cf.run_metrics(segmented_project, metrics=[cf.body_length()],
                   run_name="chosen_name", visualize=False)
    runs = run_records.load_runs(segmented_project, kind="metric")
    assert runs.iloc[0]["name"] == "chosen_name"


# ---------------------------------------------------------------------------
# requires_mask=False -- a metric that can run before segmentation
# ---------------------------------------------------------------------------


def _saw_mask():
    """A synthetic maskless metric: records whether a mask happened to be
    there, without ever requiring one -- exercises the general mechanism
    without needing usability_annotation's interactive window."""
    return Metric("saw_mask", lambda segment: segment.mask is not None,
                  version="1", unit="category", requires_mask=False)


def test_a_maskless_metric_measures_every_occurrence_with_an_image(image_project):
    """The whole point: nothing needs to be segmented first."""
    result = cf.run_metrics(image_project, metrics=[_saw_mask()],
                            visualize=False)["organism"]
    assert (result["processed"], result["skipped"]) == (8, 0)


def test_a_maskless_metric_still_sees_a_mask_where_one_already_exists(image_project):
    """Segmenting first doesn't break the maskless path -- it uses whatever's
    actually there instead of always measuring blind."""
    cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                    limit=3, visualize=False)

    cf.run_metrics(image_project, run_name="saw_mask", metrics=[_saw_mask()],
                   visualize=False)

    values = cf.load_metrics(image_project, run_names=["saw_mask"])
    assert sorted(values["value"]) == [False] * 5 + [True] * 3


def test_mixing_with_a_mask_requiring_metric_still_needs_a_mask(image_project):
    """any(), not all(): one mask-requiring metric in the list is enough to
    put the whole recipe back behind segmentation."""
    result = cf.run_metrics(image_project, run_name="mixed",
                            metrics=[_saw_mask(), cf.body_length()],
                            visualize=False)["organism"]
    assert result["processed"] == 0


def test_transforms_alone_force_a_mask_requirement(image_project):
    """A transform chain is assumed to want a mask even if every metric
    downstream of it claims otherwise."""
    result = cf.run_metrics(image_project, run_name="oriented",
                            metrics=[_saw_mask()], transforms=[cf.orient()],
                            visualize=False)["organism"]
    assert result["processed"] == 0


def test_a_maskless_metric_rerun_does_no_work(image_project):
    """Repeat-awareness still applies -- passing source_mask_hashes=None to
    completed_keys (not {}) is what makes this recognize its own prior run
    rather than reprocessing every occurrence every time."""
    cf.run_metrics(image_project, metrics=[_saw_mask()], visualize=False)
    second = cf.run_metrics(image_project, metrics=[_saw_mask()],
                            visualize=False)["organism"]
    assert (second["processed"], second["skipped"]) == (0, 8)
