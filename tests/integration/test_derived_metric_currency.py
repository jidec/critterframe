"""
A metric fit on another run's stored values -- clustering embeddings -- is
current only while the values it was fit on are.

The upstream recipe sits beside the cluster metric's hash the way its reference
population does: recorded by prepare(), compared by
`metrics.run._current_for_population`. Moving the upstream run onto a new
recipe therefore rescores the cluster run on its next pass, instead of leaving
values fit on replaced embeddings looking current.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.project import paths

pytest.importorskip("sklearn")

SPECIMENS = 8


class ColourEmbedder:
    """A torch-free stand-in for a network: mean and spread of each channel."""

    def identity(self):
        return {"class": "ColourEmbedder", "version": "1"}

    def embed(self, image):
        pixels = np.asarray(image, float).reshape(-1, 3)
        return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


def embed(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(project_path, metrics=[cf.embedding(ColourEmbedder())],
                          **kwargs)["organism"]


def clusters(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    metric = cf.cluster([cf.embedding(ColourEmbedder())], from_run="embedding",
                        n_clusters=2)
    return cf.run_metrics(project_path, run_name="groups", metrics=[metric],
                          **kwargs)["organism"]


def test_embeddings_cluster_like_any_stored_trait(segmented_project):
    assert embed(segmented_project)["processed"] == SPECIMENS
    assert clusters(segmented_project)["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, run_names=["groups"], path=False)
    assert set(exported["groups__organism__cluster__cluster_id"]) <= {0, 1}
    assert len(exported) == SPECIMENS


def test_a_cluster_run_is_repeat_aware(segmented_project):
    embed(segmented_project)
    clusters(segmented_project)
    again = clusters(segmented_project)
    assert (again["processed"], again["skipped"]) == (0, SPECIMENS)


def test_moving_the_upstream_run_to_a_new_recipe_rescores_the_clusters(
        segmented_project):
    """
    The cluster metric's own hash doesn't move -- its features are configured
    exactly as before -- and neither does its population. Only the embeddings
    underneath changed, which is what from_recipe_hash records.
    """
    embed(segmented_project)
    clusters(segmented_project)

    embed(segmented_project, transforms=[cf.crop_to_mask()], force=True)
    rescored = clusters(segmented_project)

    assert rescored["processed"] == SPECIMENS
    assert rescored["skipped"] == 0


def test_a_cluster_run_draws_its_galleries_and_projection(segmented_project):
    embed(segmented_project)
    clusters(segmented_project, visualize=True)

    written = {path.name for path in paths.pipeline_dir(segmented_project).glob("*.png")}
    assert any(name.endswith("__cluster__clusters.png") for name in written)
    assert any(name.endswith("__cluster__reference.png") for name in written)
