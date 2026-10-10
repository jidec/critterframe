"""
Clusters: a cluster id per occurrence within its own group, from stored values.

What a group metric does whatever its model is tested in `base/test_group.py`;
this is what is the cluster metric's own: the cluster count, a clusterer that
cannot predict, a probability threshold, and a PCA step in front of the model.
"""

import pytest

import critterframe as cf
from critterframe.metrics.base.stored import StoredValues
from critterframe.metrics.run_metrics import RunContext
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.core.recipes import Recipe

pytest.importorskip("sklearn")


def a_context(project_path, occurrence_ids=None):
    return RunContext(
        project_path, occurrence_ids or [f"specimen{index}" for index in range(8)], "organism", "scores"
    )


class StubEmbedder:
    """Stands in for a network: an identity for the hash, nothing to run."""

    def __init__(self, weights="a"):
        self.weights = weights

    def identity(self):
        return {"class": "StubEmbedder", "weights": self.weights}

    def embed(self, image):
        raise AssertionError("stored scoring must not rerun the network")


def two_clusters():
    """Four specimens near one point, four near another, as 3-d vectors."""
    return {f"specimen{index}": ([0.0, 0.0, 1.0] if index < 4 else [1.0, 1.0, 0.0]) for index in range(8)}


def store_vectors(project_path, values, model=None, run_name="embed"):
    recipe = Recipe("metric", run_name, [cf.embedding(model or StubEmbedder())], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [
            make_metric_row(occurrence_id, "organism", "embedding", value, unit="embedding")
            for occurrence_id, value in values.items()
        ],
    )
    return recipe.hash


def stored(occurrence_id):
    return StoredValues(occurrence_id, "organism")


def stored_cluster(**kwargs):
    kwargs.setdefault("n_clusters", 2)
    return cf.cluster([cf.embedding(StubEmbedder())], from_run="embed", **kwargs)


@pytest.mark.slow
def test_a_cluster_assignment_is_a_metric_like_any_other(measured_project):
    """
    Which is the point of metrics being "any derived value": a cluster label
    stores, exports, and filters exactly like a body length.
    """
    cf.run_metrics(
        measured_project,
        run_name="groups",
        metrics=[cf.cluster([cf.body_length()], from_run="traits", n_clusters=2)],
        visualize=False,
    )
    exported = cf.export_metrics(measured_project, run_names=["groups"])
    labels = exported["groups__organism__cluster__cluster_id"]

    assert set(labels.unique()) <= {0, 1}
    assert len(labels) == 8
    # The distance to the assigned centroid rides along: large despite being
    # the NEAREST centre still means this occurrence sits far from its peers.
    assert (exported["groups__organism__cluster__centroid_distance"] >= 0).all()


def test_a_different_cluster_count_is_different_work():
    """It used to hash alike, so asking for 5 clusters skipped as done at 3."""
    three = cf.cluster([cf.body_length()], from_run="traits", n_clusters=3)
    five = cf.cluster([cf.body_length()], from_run="traits", n_clusters=5)
    assert three.spec() != five.spec()


def test_a_clusterer_that_cannot_predict_reports_its_own_labels(metadata_project):
    """DBSCAN labels only what it was fit on, which stored scoring is; -1 is noise."""
    from sklearn.cluster import DBSCAN

    values = two_clusters()
    values["specimen7"] = [9.0, 9.0, 9.0]
    store_vectors(metadata_project, values)
    metric = stored_cluster(model_factory=lambda: DBSCAN(eps=0.5, min_samples=3))
    metric.prepare(a_context(metadata_project))

    labels = [metric(stored(f"specimen{index}"))["cluster_id"] for index in range(8)]
    assert labels[7] == -1
    assert labels[0] != labels[4] and -1 not in labels[:7]


def test_a_probability_below_the_threshold_is_unassigned(metadata_project):
    from sklearn.mixture import GaussianMixture

    store_vectors(metadata_project, two_clusters())
    metric = stored_cluster(
        model_factory=lambda: GaussianMixture(2, random_state=0, reg_covar=1e-3), probability_threshold=1.01
    )
    metric.prepare(a_context(metadata_project))

    result = metric(stored("specimen0"))
    assert result["cluster_id"] == -1
    assert 0 <= result["probability"] <= 1


def test_a_threshold_needs_a_probabilistic_model():
    with pytest.raises(ValueError, match="predict_proba"):
        cf.cluster([cf.body_length()], from_run="traits", probability_threshold=0.5)


def test_reducing_dimensions_first_is_a_pipeline_in_the_hash(metadata_project):
    store_vectors(metadata_project, two_clusters())
    reduced = stored_cluster(n_components=2)
    reduced.prepare(a_context(metadata_project))

    assert "centroid_distance" in reduced(stored("specimen0"))
    assert reduced.spec() != stored_cluster().spec()
