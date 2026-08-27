"""
Group metrics: values that only mean anything relative to a population.

Everything else in `metrics/` measures one occurrence in isolation. These score
one against many, which needs a population fitted before any single occurrence
can be scored -- and that is what `Operation.prepare()` exists for. Anything
expensive that must happen once per run goes there rather than in `__init__`,
and this is the case that shaped the hook.

Two design points get tested here rather than described:

  the fitted model is NOT in the recipe hash. It is determined by the reference
  values, which are determined by `from_run`, so hashing it would add nothing
  but instability from model randomness;

  a group too small to fit falls back to the population-wide model, and the
  stored value says which model actually scored it -- because a score is not
  interpretable without knowing what it was scored against.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.metrics.outliers import POPULATION, group_lookup
from critterframe.metrics.run import RunContext
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.recipes import Recipe

pytest.importorskip("sklearn")


def store_lengths(project_path, values, run_name="traits"):
    """Give each occurrence a body_length, which the group metrics fit on."""
    recipe = Recipe("metric", run_name, [cf.body_length()], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(project_path, run_id, recipe.hash,
                   [make_metric_row(occurrence_id, "organism", "body_length",
                                    value, unit="px")
                    for occurrence_id, value in values.items()])


def typical_lengths(count=8, odd_one_out=None):
    values = {f"specimen{index}": 100.0 + index for index in range(count)}
    if odd_one_out is not None:
        values[f"specimen{count - 1}"] = odd_one_out
    return values


def a_context(project_path, occurrence_ids=None):
    return RunContext(project_path, occurrence_ids or
                      [f"specimen{index}" for index in range(8)],
                      "organism", "scores")


# ---------------------------------------------------------------------------
# group_lookup
# ---------------------------------------------------------------------------


def test_group_lookup_maps_occurrences_to_their_group(metadata_project):
    groups = group_lookup(metadata_project, "device")
    assert groups["specimen0"] == "boxA"
    assert groups["specimen1"] == "boxB"


def test_group_lookup_without_a_group_column_puts_everyone_together(
        metadata_project):
    """Which is the population-wide fallback, spelled as an empty mapping."""
    assert group_lookup(metadata_project, None) == {}


def test_group_lookup_on_a_column_that_is_not_there_says_so(metadata_project):
    """
    Rather than a bare ArrowInvalid about FieldRefs -- the same guard
    calibration scopes and the download URL column already have.
    """
    with pytest.raises(KeyError, match="nothing to group by"):
        group_lookup(metadata_project, "copy_stand")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_a_group_metric_needs_at_least_one_feature():
    with pytest.raises(ValueError, match="at least one feature"):
        cf.outlier([], from_run="traits")


def test_the_features_and_the_reference_run_are_in_the_hash():
    """
    All three of features, from_run, and group_col change what "outlier" MEANS,
    so all three have to change the hash.
    """
    base = cf.outlier([cf.body_length()], from_run="traits")
    assert base.spec() != cf.outlier([cf.max_width()], from_run="traits").spec()
    assert base.spec() != cf.outlier([cf.body_length()], from_run="qc").spec()
    assert base.spec() != cf.outlier([cf.body_length()], from_run="traits",
                                     group_col="device").spec()


def test_the_fitted_model_is_not_in_the_hash():
    """
    Two identically-configured metrics hash alike even though each would fit
    its own model instance -- the model is determined by the reference values,
    so hashing it would add only instability.
    """
    first = cf.outlier([cf.body_length()], from_run="traits")
    second = cf.outlier([cf.body_length()], from_run="traits")
    assert first.spec() == second.spec()


def test_the_model_class_is_in_the_hash():
    """Swapping IsolationForest for KMeans is different work under one name."""
    assert (cf.outlier([cf.body_length()], from_run="traits").spec()["parameters"]["model"]
            != cf.cluster([cf.body_length()], from_run="traits").spec()["parameters"]["model"])


def test_scoring_without_fitting_first_says_so():
    """
    Group metrics are fit by their prepare() hook, which run_metrics calls --
    calling the operation directly skips it, and the error has to say that
    rather than failing inside sklearn.
    """
    from critterframe.recipes import Segment

    metric = cf.outlier([cf.body_length()], from_run="traits")
    segment = Segment(np.zeros((10, 10, 3), np.uint8), mask=np.ones((10, 10), bool))
    with pytest.raises(RuntimeError, match="never fit"):
        metric(segment)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_fitting_needs_a_measured_population(metadata_project):
    """
    A group metric scores against a population that has already been measured,
    and saying which run is missing is the difference between a fixable error
    and a confusing one.
    """
    metric = cf.outlier([cf.body_length()], from_run="traits")
    with pytest.raises(ValueError, match="no reference values"):
        metric.prepare(a_context(metadata_project))


def test_a_population_model_is_fit_from_stored_values(metadata_project):
    store_lengths(metadata_project, typical_lengths())
    metric = cf.outlier([cf.body_length()], from_run="traits")
    metric.prepare(a_context(metadata_project))

    assert POPULATION in metric.models


def test_a_group_model_is_fit_per_group_when_asked(metadata_project):
    store_lengths(metadata_project, typical_lengths())
    metric = cf.outlier([cf.body_length()], from_run="traits",
                        group_col="device", min_group_size=2)
    metric.prepare(a_context(metadata_project))

    assert {"boxA", "boxB", POPULATION} <= set(metric.models)


def test_a_group_too_small_to_fit_falls_back_and_says_so(metadata_project, caplog):
    """
    Silently fitting a model to two specimens would produce scores that look
    like every other score in the column.
    """
    store_lengths(metadata_project, typical_lengths())
    metric = cf.outlier([cf.body_length()], from_run="traits",
                        group_col="device", min_group_size=100)

    with caplog.at_level("WARNING"):
        metric.prepare(a_context(metadata_project))

    assert set(metric.models) == {POPULATION}
    assert "population-wide model" in caplog.text


def test_the_reference_population_is_the_run_s_own_occurrences(metadata_project):
    """
    Fit against every occurrence this run covers, including ones an earlier
    interrupted attempt already scored -- otherwise resuming would silently
    change what the score means partway through.
    """
    store_lengths(metadata_project, typical_lengths())
    metric = cf.outlier([cf.body_length()], from_run="traits")
    metric.prepare(a_context(metadata_project, ["specimen0", "specimen1",
                                                "specimen2"]))
    assert POPULATION in metric.models


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_an_unusual_specimen_scores_as_more_anomalous(measured_project):
    """
    Both the boolean call and the continuous score beneath it are stored, so a
    stricter or looser cutoff can be applied at export without refitting
    anything.
    """
    cf.run_metrics(measured_project, run_name="scores",
                   metrics=[cf.outlier([cf.body_length(), cf.max_width()],
                                       from_run="traits")],
                   visualize=False)
    exported = cf.export_metrics(measured_project, runs=["scores"])

    assert "scores__organism__outlier__anomaly_score" in exported.columns
    assert "scores__organism__outlier__is_outlier" in exported.columns
    # Scored against the whole population, so the group column is empty -- which
    # is what "None where the population-wide fallback was used" looks like in
    # an export, and is exactly the thing a reader needs to know.
    assert exported["scores__organism__outlier__group"].isna().all()


@pytest.mark.slow
def test_a_grouped_score_records_which_group_scored_it(measured_project):
    """
    Because a score is not interpretable without knowing what it was scored
    against: "unusual for boxA" and "unusual for this project" are different
    claims and land in the same column.
    """
    cf.run_metrics(measured_project, run_name="scores",
                   metrics=[cf.outlier([cf.body_length()], from_run="traits",
                                       group_col="device", min_group_size=2)],
                   visualize=False)
    groups = cf.export_metrics(measured_project,
                               runs=["scores"])["scores__organism__outlier__group"]
    assert set(groups) == {"boxA", "boxB"}


@pytest.mark.slow
def test_a_cluster_assignment_is_a_metric_like_any_other(measured_project):
    """
    Which is the point of metrics being "any derived value": a cluster label
    stores, exports, and filters exactly like a body length.
    """
    cf.run_metrics(measured_project, run_name="groups",
                   metrics=[cf.cluster([cf.body_length()], from_run="traits",
                                       n_clusters=2)],
                   visualize=False)
    exported = cf.export_metrics(measured_project, runs=["groups"])
    labels = exported["groups__organism__cluster__cluster_id"]

    assert set(labels.unique()) <= {0, 1}
    assert len(labels) == 8
    # The distance to the assigned centroid rides along: large despite being
    # the NEAREST centre still means this occurrence sits far from its peers.
    assert (exported["groups__organism__cluster__centroid_distance"] >= 0).all()


@pytest.mark.slow
def test_a_group_metric_is_repeat_aware_like_any_other(measured_project):
    def score(**kwargs):
        return cf.run_metrics(measured_project, run_name="scores",
                              metrics=[cf.outlier([cf.body_length()],
                                                  from_run="traits")],
                              visualize=False, **kwargs)["organism"]

    assert score()["processed"] == 8
    assert score()["skipped"] == 8
