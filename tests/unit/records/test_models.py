"""
Registered models: provenance about weights, and how it reaches a recipe hash.

Training happens outside the package -- this module imports no framework and
loads nothing. What it records is the join between a checkpoint and the data
behind it, and the one thing that must be exactly right is what goes into
`identity()`:

  the FINGERPRINT is in, because retraining into the same filename has to move
  the recipe hash of everything the model produces, or a project keeps serving
  masks from weights it no longer has;

  the PATH is out, because two copies of one checkpoint are the same model and
  moving a project must not invalidate every mask in it.
"""

import json

import pytest

from critterframe.project import paths
from critterframe.records import models as model_records
from critterframe.records.models import RegisteredModel, register_model
from critterframe.recipes import Recipe, model_identity
from critterframe.segmentation.run import segment
from helpers.models import ThresholdModel


@pytest.fixture
def checkpoint(metadata_project):
    """A stand-in for weights trained elsewhere, inside the project."""
    path = paths.models_dir(metadata_project) / "blobnet_v1.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pretend weights, revision 1")
    return path


def registry_file(project_path):
    with paths.models_registry_path(project_path).open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Registering
# ---------------------------------------------------------------------------


def test_a_registration_records_what_the_weights_are(metadata_project, checkpoint):
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint,
                           task="segment", framework="torch",
                           base_model="sam2_hiera_large",
                           parameters={"epochs": 40})

    assert model.name == "blobnet_v1"
    assert model.record["task"] == "segment"
    assert model.record["base_model"] == "sam2_hiera_large"
    assert model.record["parameters"] == {"epochs": 40}
    assert model.record["fingerprint"]


def test_the_registry_is_a_readable_file(metadata_project, checkpoint):
    register_model(metadata_project, "blobnet_v1", path=checkpoint)
    assert "blobnet_v1" in registry_file(metadata_project)["models"]


def test_a_checkpoint_inside_the_project_is_stored_relative(metadata_project,
                                                            checkpoint):
    """
    A project is meant to be copyable, and an absolute path baked into its
    registry is the thing that breaks first when it moves to a cluster.
    """
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    assert model.record["path"] == "models/blobnet_v1.pt"
    assert model.path == checkpoint


def test_a_checkpoint_elsewhere_keeps_its_absolute_path(metadata_project, tmp_path):
    outside = tmp_path / "shared" / "weights.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"weights")

    model = register_model(metadata_project, "shared_v1", path=outside)
    assert model.path.resolve() == outside.resolve()


def test_a_missing_checkpoint_raises(metadata_project):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        register_model(metadata_project, "ghost", path="models/nothing.pt")


@pytest.mark.parametrize("name", ["bad name", "with/slash", "", ".hidden"])
def test_a_name_that_cannot_be_a_key_or_a_filename_raises(metadata_project,
                                                          checkpoint, name):
    with pytest.raises(ValueError, match="model name"):
        register_model(metadata_project, name, path=checkpoint)


def test_a_directory_checkpoint_is_fingerprinted_as_a_whole(metadata_project):
    """
    A checkpoint saved as a folder of shards is identified as precisely as a
    single file -- a shard appearing or moving inside it changes the answer.
    """
    directory = paths.models_dir(metadata_project) / "sharded"
    directory.mkdir(parents=True)
    (directory / "part1.bin").write_bytes(b"aaa")
    (directory / "part2.bin").write_bytes(b"bbb")

    first = register_model(metadata_project, "sharded_v1", path=directory)
    (directory / "part3.bin").write_bytes(b"ccc")
    second = register_model(metadata_project, "sharded_v1", path=directory)

    assert first.record["fingerprint"] != second.record["fingerprint"]


def test_a_model_with_no_local_weights_is_allowed_and_warned_about(
        metadata_project, caplog):
    """
    A hosted endpoint. Identity then rests on the name alone, and saying so is
    the difference between a known limitation and a silent one.
    """
    with caplog.at_level("WARNING"):
        model = register_model(metadata_project, "hosted", task="embedding")

    assert model.identity()["fingerprint"] is None
    assert model.path is None
    assert "identity rests on its name alone" in caplog.text


def test_fingerprinting_can_be_declined_loudly(metadata_project, checkpoint,
                                               caplog):
    with caplog.at_level("WARNING"):
        model = register_model(metadata_project, "big_v1", path=checkpoint,
                               fingerprint=False)
    assert model.record["fingerprint"] is None
    assert "will not change any recipe hash" in caplog.text


# ---------------------------------------------------------------------------
# identity()
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_the_identity(metadata_project, checkpoint):
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    identity = model.identity()

    assert identity["fingerprint"] == model.record["fingerprint"]
    assert "path" not in identity


def test_retraining_into_the_same_filename_moves_the_hash(metadata_project,
                                                          checkpoint, caplog):
    """
    THE failure this exists to prevent. Same name, same path, same class --
    different weights, so every mask and metric derived from it is correctly
    redone rather than silently kept.
    """
    before = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    before_hash = Recipe("segment", "custom",
                         [segment(before.attach(ThresholdModel()))]).hash

    checkpoint.write_bytes(b"pretend weights, revision 2 -- trained for longer")
    with caplog.at_level("WARNING"):
        after = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    after_hash = Recipe("segment", "custom",
                        [segment(after.attach(ThresholdModel()))]).hash

    assert before_hash != after_hash
    assert "re-registered with different weights" in caplog.text


def test_the_same_weights_under_two_names_are_two_models(metadata_project,
                                                         checkpoint):
    """
    The name is in identity() as well as the fingerprint: registering the same
    file twice is a deliberate act, and the two names may carry different
    training provenance.
    """
    first = register_model(metadata_project, "a_v1", path=checkpoint)
    second = register_model(metadata_project, "b_v1", path=checkpoint)

    assert first.record["fingerprint"] == second.record["fingerprint"]
    assert first.identity() != second.identity()


def test_moving_the_project_does_not_change_the_identity(metadata_project,
                                                         checkpoint, tmp_path):
    """
    Two copies of one checkpoint are the same model. If the path were in the
    identity, copying a project to a cluster would invalidate every mask in it.
    """
    import shutil

    original = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    copied_project = tmp_path / "copied"
    shutil.copytree(metadata_project, copied_project)

    copied = model_records.load_model(copied_project, "blobnet_v1")
    assert copied.identity() == original.identity()
    assert copied.path.exists()


def test_a_registered_model_reaches_a_recipe_hash_like_any_other(metadata_project,
                                                                  checkpoint):
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    assert model_identity(model) == model.identity()


# ---------------------------------------------------------------------------
# attach()
# ---------------------------------------------------------------------------


def test_attaching_binds_a_loaded_network_without_changing_the_record(
        metadata_project, checkpoint):
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint)
    attached = model.attach(ThresholdModel())

    assert attached.identity() == model.identity()
    assert attached.runtime is not None
    assert model.runtime is None            # a new object, not a mutation


def test_an_attached_model_forwards_the_work_to_the_network(metadata_project,
                                                            checkpoint,
                                                            draw_specimen):
    """
    Which is what makes a registered model usable anywhere a model is: the
    registry answers identity(), the network answers predict().
    """
    import cv2

    model = register_model(metadata_project, "blobnet_v1",
                           path=checkpoint).attach(ThresholdModel())
    mask, score, _info = model.predict(cv2.cvtColor(draw_specimen(0),
                                                    cv2.COLOR_BGR2RGB))
    assert mask.any() and score == 0.9


def test_an_unattached_model_has_no_methods_to_offer(metadata_project, checkpoint):
    """
    AttributeError rather than something louder, because hasattr() is how a run
    asks what a model can do -- a check for visualize() must get False, not an
    exception.
    """
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint)

    assert hasattr(model, "visualize") is False
    with pytest.raises(AttributeError, match="no loaded network"):
        model.predict(None)


def test_hasattr_reflects_what_the_attached_network_actually_has(metadata_project,
                                                                 checkpoint):
    model = register_model(metadata_project, "blobnet_v1",
                           path=checkpoint).attach(ThresholdModel())
    assert hasattr(model, "predict") is True
    assert hasattr(model, "visualize") is False


# ---------------------------------------------------------------------------
# Training provenance
# ---------------------------------------------------------------------------


def test_training_splits_are_recorded_as_counts_and_digests(metadata_project,
                                                            checkpoint):
    """
    The point is to be able to prove which set was used, which a digest does --
    and thousands of ids in a registry file would make it unreadable for no
    gain.
    """
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint,
                           training_splits={"train": ["specimen0", "specimen1",
                                                      "specimen2"],
                                            "val": ["specimen3"]})
    splits = model.record["training_data"]["splits"]

    assert splits["train"]["count"] == 3
    assert splits["val"]["ids_hash"]
    assert "specimen" not in json.dumps(splits)


@pytest.mark.slow
def test_an_exported_dataset_can_be_pointed_at(segmented_project, tmp_path):
    """
    The common path: export_training_data writes a dataset.json holding the
    split digests, the part, and the transform chain, so pointing at the
    directory carries the whole answer.
    """
    import critterframe as cf

    out = tmp_path / "dataset"
    cf.export_training_data(segmented_project, out,
                            splits={"train": [f"specimen{i}" for i in range(6)],
                                    "val": ["specimen6", "specimen7"]},
                            masks=True)

    weights = paths.models_dir(segmented_project) / "seg_v1.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")

    model = register_model(segmented_project, "seg_v1", path=weights,
                           training_data=out)
    dataset = model.record["training_data"]["dataset"]

    assert dataset["splits"]["train"]["count"] == 6
    assert dataset["part"] == "organism"
    assert dataset["data_hash"]


def test_a_dataset_json_can_be_named_directly(metadata_project, checkpoint,
                                              tmp_path):
    record = {"data_hash": "abc", "splits": {"train": {"count": 2}}}
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps(record), encoding="utf-8")

    model = register_model(metadata_project, "blobnet_v1", path=checkpoint,
                           training_data=dataset_json)
    assert model.record["training_data"]["dataset"]["data_hash"] == "abc"


def test_pointing_at_a_dataset_that_is_not_there_raises(metadata_project,
                                                        checkpoint, tmp_path):
    with pytest.raises(FileNotFoundError, match="no dataset record"):
        register_model(metadata_project, "blobnet_v1", path=checkpoint,
                       training_data=tmp_path / "nowhere")


def test_training_settings_are_stored_and_never_interpreted(metadata_project,
                                                            checkpoint):
    """
    Exactly as a calibration's parameters are: what matters varies per
    framework, and flattening it would fit one and distort the rest.
    """
    parameters = {"epochs": 40, "loss": {"name": "arcface", "margin": 0.3},
                  "augmentations": ["hflip"]}
    model = register_model(metadata_project, "blobnet_v1", path=checkpoint,
                           parameters=parameters)
    assert model.record["parameters"] == parameters


# ---------------------------------------------------------------------------
# Reading the registry
# ---------------------------------------------------------------------------


def test_a_project_with_no_models_is_not_an_error(metadata_project):
    assert model_records.list_models(metadata_project) == {}


def test_loading_a_model_nobody_registered_says_what_is_there(metadata_project,
                                                              checkpoint):
    register_model(metadata_project, "blobnet_v1", path=checkpoint)
    with pytest.raises(KeyError, match="no model named 'blobnet_v2'"):
        model_records.load_model(metadata_project, "blobnet_v2")


def test_loading_gives_provenance_and_no_runtime(metadata_project, checkpoint):
    register_model(metadata_project, "blobnet_v1", path=checkpoint)
    loaded = model_records.load_model(metadata_project, "blobnet_v1")

    assert isinstance(loaded, RegisteredModel)
    assert loaded.runtime is None
    assert loaded.record["fingerprint"]


def test_unregistering_forgets_the_record_and_nothing_else(metadata_project,
                                                           checkpoint):
    """
    The checkpoint stays on disk, and so do the masks and metrics it produced --
    they remain valid results of a recipe whose hash still names those weights.
    """
    register_model(metadata_project, "blobnet_v1", path=checkpoint)
    model_records.unregister_model(metadata_project, "blobnet_v1")

    assert model_records.list_models(metadata_project) == {}
    assert checkpoint.exists()


def test_unregistering_something_that_is_not_there_raises(metadata_project):
    with pytest.raises(KeyError, match="no model named"):
        model_records.unregister_model(metadata_project, "ghost")
