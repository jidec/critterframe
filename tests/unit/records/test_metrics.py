"""
The metric log: append-only history, and which of it is current.

The invariant under test is the one that makes the history safe to keep: a
metric row is an immutable historical result, and a value is *current* only
while the mask it was measured from still is. `current_rows` is where that
judgement is made, and it is deliberately conservative in two directions --
unrecorded provenance is unjudgeable rather than stale, and a project with no
mask table has nothing to judge against. Getting either backwards would throw
away real values.
"""

import pandas as pd
import pytest

from critterframe.recipes import Recipe
from critterframe.records import masks as mask_records
from critterframe.records import metrics as metric_records
from critterframe.records import runs as run_records
from critterframe.metrics.dimensions import body_length
from helpers.synthetic import blob_mask


def a_run(project_path, name="traits"):
    """Start a run and return (run_id, recipe_hash) to append values under."""
    recipe = Recipe("metric", name, [body_length()], part="organism")
    return run_records.start_run(project_path, recipe), recipe.hash


def store(project_path, values, run_name="traits", recipe_hash=None,
          source_mask_hash=None, part="organism", metric_name="body_length"):
    """Append {occurrence_id: value} under one run, returning the run_id."""
    run_id, hash_of_recipe = a_run(project_path, run_name)
    metric_records.append_metrics(
        project_path, run_id, recipe_hash or hash_of_recipe,
        [metric_records.make_metric_row(occurrence_id, part, metric_name, value,
                                        unit="px",
                                        source_mask_hash=source_mask_hash)
         for occurrence_id, value in values.items()])
    return run_id


def save_mask(project_path, occurrence_id="a", part="organism",
              recipe_hash="seg_v1", **kwargs):
    mask_records.save_masks(project_path, [
        mask_records.make_mask_row(occurrence_id, blob_mask(), part=part,
                                   recipe_hash=recipe_hash, **kwargs)])


# ---------------------------------------------------------------------------
# make_metric_row / append_metrics
# ---------------------------------------------------------------------------


def test_ids_are_strings_on_the_row():
    row = metric_records.make_metric_row(7, "organism", "body_length", 1.0)
    assert row["occurrence_id"] == "7"


def test_a_row_stores_nothing_the_run_already_records():
    """
    No per-row version (it is in the recipe spec, hence in the recipe hash) and
    no per-row timestamp (the run has one). A second copy of provenance is free
    to drift and not obviously authoritative when it does.
    """
    row = metric_records.make_metric_row("a", "organism", "body_length", 1.0)
    assert set(row) == {"occurrence_id", "part", "metric_name", "value", "unit",
                        "source_mask_hash"}


def test_appending_nothing_is_not_an_error(tmp_path):
    run_id, recipe_hash = a_run(tmp_path)
    assert metric_records.append_metrics(tmp_path, run_id, recipe_hash, []) == 0


@pytest.mark.parametrize("value", [
    12.5, 0, -3, True, None, "usable",
    [1.0, 2.0, 3.0],
    {"x": 1, "y": 2},
    {"nested": {"a": [1, 2]}},
])
def test_any_json_shaped_value_round_trips(tmp_path, value):
    """
    A metric is any derived value: a trait, a QC score, a human label, a
    128-float embedding, a cluster assignment. Values are stored as JSON
    precisely so a new metric shape needs no schema change.
    """
    store(tmp_path, {"a": value})
    assert metric_records.load_metrics(tmp_path)["value"].iloc[0] == value


def test_values_carry_their_run_and_recipe(tmp_path):
    """
    The run's name, kind, and time come along on the read, which is why no row
    stores its own copy.
    """
    store(tmp_path, {"a": 1.0}, run_name="traits")
    row = metric_records.load_metrics(tmp_path).iloc[0]
    assert row["run_name"] == "traits"
    assert row["run_kind"] == "metric"
    assert row["unit"] == "px"
    assert row["recipe_hash"]


def test_load_filters_by_run_part_and_metric(tmp_path):
    store(tmp_path, {"a": 1.0}, run_name="traits")
    store(tmp_path, {"a": 2.0}, run_name="qc", metric_name="blur_variance")
    store(tmp_path, {"a": 3.0}, run_name="traits", part="wing")

    assert len(metric_records.load_metrics(tmp_path, run_names=["qc"])) == 1
    assert len(metric_records.load_metrics(tmp_path, parts=["wing"])) == 1
    assert len(metric_records.load_metrics(
        tmp_path, metric_names=["body_length"])) == 2


def test_an_empty_log_still_has_the_right_columns(tmp_path):
    """
    Callers reshape and filter this frame. An empty one without columns would
    make every one of them fail on the empty case rather than return nothing.
    """
    run_records.start_run(tmp_path, Recipe("metric", "traits", [body_length()]))
    empty = metric_records.load_metrics(tmp_path)
    assert empty.empty
    assert {"occurrence_id", "part", "metric_name", "value", "unit",
            "recipe_hash", "source_mask_hash"} <= set(empty.columns)


# ---------------------------------------------------------------------------
# current_rows -- which history still describes the project
# ---------------------------------------------------------------------------


def test_a_value_measured_from_the_current_mask_is_current(tmp_path):
    save_mask(tmp_path, recipe_hash="seg_v1")
    store(tmp_path, {"a": 1.0}, source_mask_hash="seg_v1")

    long_df = metric_records.load_metrics(tmp_path)
    assert len(metric_records.current_rows(tmp_path, long_df)) == 1


def test_a_value_measured_from_a_replaced_mask_is_not(tmp_path):
    """
    THE staleness rule. Resegmenting deletes nothing -- it just means the values
    derived from the old mask stop being current, and the long table
    legitimately holds both.
    """
    store(tmp_path, {"a": 1.0}, source_mask_hash="seg_v1")
    save_mask(tmp_path, recipe_hash="seg_v2")

    long_df = metric_records.load_metrics(tmp_path)
    assert len(long_df) == 1
    assert metric_records.current_rows(tmp_path, long_df).empty


def test_a_value_of_unrecorded_provenance_is_kept(tmp_path):
    """
    Unjudgeable, not known-stale. Values written before source_mask_hash
    existed have no claim to check, and discarding them would silently empty
    an older project's export.
    """
    save_mask(tmp_path, recipe_hash="seg_v2")
    store(tmp_path, {"a": 1.0}, source_mask_hash=None)

    long_df = metric_records.load_metrics(tmp_path)
    assert len(metric_records.current_rows(tmp_path, long_df)) == 1


def test_everything_is_kept_when_the_project_has_no_masks(tmp_path):
    """"No masks" must not mean "no values"."""
    store(tmp_path, {"a": 1.0}, source_mask_hash="seg_v1")
    long_df = metric_records.load_metrics(tmp_path)
    assert len(metric_records.current_rows(tmp_path, long_df)) == 1


def test_staleness_is_judged_per_occurrence_part(tmp_path):
    """
    One resegmented occurrence doesn't invalidate its neighbours, and one
    resegmented part doesn't invalidate the others of the same occurrence.
    """
    store(tmp_path, {"a": 1.0, "b": 2.0}, source_mask_hash="seg_v1")
    save_mask(tmp_path, "a", recipe_hash="seg_v2")   # a moved
    save_mask(tmp_path, "b", recipe_hash="seg_v1")   # b did not

    current = metric_records.current_rows(tmp_path,
                                          metric_records.load_metrics(tmp_path))
    assert current["occurrence_id"].tolist() == ["b"]


def test_a_reference_mask_keeps_its_own_values_current(tmp_path):
    """
    The long table doesn't record which mask table a run measured, so the two
    are pooled. A value derived from a reference mask must not be discarded for
    failing to match a canonical mask it was never derived from.
    """
    save_mask(tmp_path, recipe_hash="auto_v2")
    mask_records.save_masks(tmp_path, [
        mask_records.make_mask_row("a", blob_mask(), recipe_hash="human_v1")],
        reference=True)
    store(tmp_path, {"a": 1.0}, source_mask_hash="human_v1")

    current = metric_records.current_rows(tmp_path,
                                          metric_records.load_metrics(tmp_path))
    assert len(current) == 1


def test_current_rows_of_an_empty_frame_is_empty(tmp_path):
    empty = pd.DataFrame(columns=["occurrence_id", "part", "source_mask_hash"])
    assert metric_records.current_rows(tmp_path, empty).empty


def test_current_rows_follows_a_derived_part_upstream(tmp_path):
    """
    A wing value's source hash is the CHAINED hash, so resegmenting the organism
    the wing was cut from makes it stale without the wing recipe changing at all.
    """
    chained = mask_records.derivation_hash("wing_v1", "organism_v1")
    save_mask(tmp_path, part="wing", recipe_hash="wing_v1",
              source_mask_hash="organism_v1")
    store(tmp_path, {"a": 1.0}, part="wing", source_mask_hash=chained)

    long_df = metric_records.load_metrics(tmp_path)
    assert len(metric_records.current_rows(tmp_path, long_df)) == 1

    save_mask(tmp_path, part="wing", recipe_hash="wing_v1",
              source_mask_hash="organism_v2")
    assert metric_records.current_rows(tmp_path, long_df).empty


# ---------------------------------------------------------------------------
# latest_values
# ---------------------------------------------------------------------------


def test_latest_values_takes_the_newest_per_occurrence(tmp_path):
    """
    Newest means last written: metric_id is insertion order, which ranks two
    values computed inside one run as well as two computed years apart.
    """
    store(tmp_path, {"a": 1.0, "b": 2.0})
    store(tmp_path, {"a": 9.0})

    values = metric_records.latest_values(tmp_path, "traits",
                                          metric_name="body_length")
    assert values["a"] == 9.0
    assert values["b"] == 2.0
    assert values.name == "body_length"


def test_latest_values_ignores_stale_values_by_default(tmp_path):
    """
    On by default, and with more at stake than in an export: a group metric fits
    a reference population from these, so a stale value shifts the distribution
    every other occurrence is scored against.
    """
    store(tmp_path, {"a": 1.0}, source_mask_hash="seg_v1")
    save_mask(tmp_path, "a", recipe_hash="seg_v2")

    assert metric_records.latest_values(tmp_path, "traits",
                                        metric_name="body_length").empty
    assert len(metric_records.latest_values(tmp_path, "traits",
                                            metric_name="body_length",
                                            current_only=False)) == 1


def test_latest_values_needs_a_metric_name(tmp_path):
    with pytest.raises(ValueError, match="needs a metric_name"):
        metric_records.latest_values(tmp_path, "traits")


def test_latest_values_of_nothing_is_an_empty_series(tmp_path):
    empty = metric_records.latest_values(tmp_path, "traits",
                                         metric_name="body_length")
    assert empty.empty
    assert empty.name == "body_length"
