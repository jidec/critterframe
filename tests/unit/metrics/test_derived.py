"""
Derived metrics: a value from one occurrence-part's own stored values.

What's pinned here is the division of labour with group metrics: a derived
function sees exactly one occurrence's values and never a segment, and its
record names no population, since who else is in scope can't change its value.
"""

import pytest

import critterframe as cf
from critterframe.core.drivers import NoInput
from critterframe.metrics.run_metrics import RunContext
from critterframe.metrics.base.stored import StoredValues
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.core.recipes import Recipe

IDS = [f"specimen{index}" for index in range(8)]


def width_ratio(values):
    return values["max_width"] / values["body_length"]


def store_traits(project_path, skip=()):
    recipe = Recipe("metric", "traits", [cf.body_length(), cf.max_width()], part="organism")
    run_id = start_run(project_path, recipe)
    rows = []
    for index, occurrence_id in enumerate(IDS):
        if occurrence_id in skip:
            continue
        rows.append(make_metric_row(occurrence_id, "organism", "body_length", 100.0 + index, unit="px"))
        rows.append(make_metric_row(occurrence_id, "organism", "max_width", 20.0, unit="px"))
    append_metrics(project_path, run_id, recipe.hash, rows)
    return recipe.hash


def a_context(project_path):
    return RunContext(project_path, IDS, "organism", "shape")


def ratio(**kwargs):
    return cf.derived(width_ratio, [cf.body_length(), cf.max_width()], from_run="traits", **kwargs)


def test_a_derived_value_is_its_function_of_the_stored_values(metadata_project):
    store_traits(metadata_project)
    metric = ratio()
    metric.prepare(a_context(metadata_project))

    assert metric(StoredValues("specimen0", "organism")) == pytest.approx(0.2)
    assert metric.metric_name == "width_ratio"


def test_the_function_sees_one_occurrence_and_nothing_else(metadata_project):
    seen = []

    store_traits(metadata_project)
    metric = ratio()
    metric.fn = seen.append
    metric.prepare(a_context(metadata_project))
    metric(StoredValues("specimen3", "organism"))

    assert seen == [{"body_length": 103.0, "max_width": 20.0}]


def test_a_derived_metric_is_never_handed_a_segment(metadata_project):
    import numpy as np

    from critterframe.core.recipes import Segment

    store_traits(metadata_project)
    metric = ratio()
    metric.prepare(a_context(metadata_project))
    assert metric.input == "stored"
    with pytest.raises(TypeError, match="StoredValues"):
        metric(Segment(np.zeros((4, 4, 3), np.uint8), occurrence_id="specimen0"))


def test_a_missing_value_is_no_input_yet(metadata_project):
    store_traits(metadata_project, skip={"specimen7"})
    metric = ratio()
    metric.prepare(a_context(metadata_project))
    with pytest.raises(NoInput):
        metric(StoredValues("specimen7", "organism"))


def test_its_record_names_the_upstream_recipe_and_no_population(metadata_project):
    """Who else is in scope can't change a per-occurrence value, so it isn't recorded."""
    upstream = store_traits(metadata_project)
    record = ratio().prepare(a_context(metadata_project))

    assert record["from_recipe_hash"] == upstream
    assert "population" not in record


def test_a_lambda_has_no_name_to_hash():
    with pytest.raises(ValueError, match="module-level"):
        cf.derived(lambda values: 1, [cf.body_length()], from_run="traits")


def test_a_nested_function_has_no_name_to_hash():
    def local(values):
        return 1

    with pytest.raises(ValueError, match="module-level"):
        cf.derived(local, [cf.body_length()], from_run="traits")


def test_the_function_and_its_version_are_in_the_hash():
    spec = ratio().spec()
    assert spec["parameters"]["function"].endswith("test_derived.width_ratio")
    assert ratio(version="2").spec() != spec
    assert cf.derived(width_ratio, [cf.body_length()], from_run="traits").spec() != spec


def scaled(values, factor):
    return values["body_length"] * factor


def test_parameters_reach_the_function_and_the_hash(metadata_project):
    store_traits(metadata_project)
    metric = cf.derived(scaled, [cf.body_length()], from_run="traits", parameters={"factor": 2})
    metric.prepare(a_context(metadata_project))

    assert metric(StoredValues("specimen1", "organism")) == pytest.approx(202.0)
    assert (
        metric.spec()
        != cf.derived(scaled, [cf.body_length()], from_run="traits", parameters={"factor": 3}).spec()
    )


def test_no_parameters_leaves_the_spec_as_it_was():
    """A derived hash recorded before parameters existed must not move."""
    assert "arguments" not in ratio().spec()["parameters"]
    assert ratio().spec() == ratio(parameters={}).spec()


def test_parameters_must_be_json():
    with pytest.raises(TypeError, match="JSON"):
        cf.derived(scaled, [cf.body_length()], parameters={"factor": object()})


def test_with_no_from_run_it_reads_this_runs_earlier_values(metadata_project):
    """Nothing is loaded: the values arrive with the occurrence."""
    metric = cf.derived(width_ratio, [cf.body_length(), cf.max_width()])
    record = metric.prepare(a_context(metadata_project))

    assert record["from_run"] is None
    values = {"body_length": 50.0, "max_width": 10.0, "mask_area": 1.0}
    assert metric(StoredValues("specimen0", "organism", values)) == pytest.approx(0.2)
    with pytest.raises(NoInput, match="earlier in this run"):
        metric(StoredValues("specimen0", "organism", {"body_length": 50.0}))


def test_a_named_feature_binds_to_the_earlier_metric_of_that_name():
    """Binding fills in what passing the operation does, so the recipe hash is the same."""
    named = cf.derived(width_ratio, ["body_length", "max_width"])
    bound = named.bind([cf.mask_area(), cf.body_length(), cf.max_width()])

    assert bound.spec() == cf.derived(width_ratio, [cf.body_length(), cf.max_width()]).spec()
    values = {"body_length": 50.0, "max_width": 10.0}
    assert bound(StoredValues("specimen0", "organism", values)) == pytest.approx(0.2)


def test_binding_leaves_the_named_metric_as_it_was():
    """One instance reused in two runs is bound to each run's own metrics."""
    named = cf.derived(scaled, ["body_length"], parameters={"factor": 2})
    in_px = named.bind([cf.body_length()])
    in_mm = named.bind([cf.body_length(unit="mm")])

    assert named.unbound == ["body_length"]
    assert in_px is not named and in_px.spec() != in_mm.spec()
    assert in_px.bind([]) is in_px


def test_a_named_metric_is_not_a_recipe_until_bound(metadata_project):
    named = cf.derived(width_ratio, ["body_length", "max_width"])
    with pytest.raises(ValueError, match="run_metrics"):
        named.spec()
    with pytest.raises(ValueError, match="run_metrics"):
        named.prepare(a_context(metadata_project))
    with pytest.raises(ValueError, match="run_metrics"):
        named(StoredValues("specimen0", "organism", {"body_length": 1.0, "max_width": 1.0}))


def test_a_name_nothing_earlier_carries_is_refused():
    named = cf.derived(width_ratio, ["body_length", "max_width"])
    with pytest.raises(ValueError, match="listed before it"):
        named.bind([cf.body_length()])


def test_another_runs_features_cannot_be_named():
    with pytest.raises(ValueError, match="pass the operations"):
        cf.derived(width_ratio, ["body_length", "max_width"], from_run="traits")


def test_a_group_metric_cannot_read_its_own_run():
    with pytest.raises(ValueError, match="needs from_run"):
        cf.outlier([cf.body_length()], from_run=None)
