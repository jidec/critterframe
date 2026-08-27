"""
The whole path from a project to a model that measures it.

    reference masks
      -> split_ids()
      -> export_training_data()
      -> [training happens outside CritterFrame]
      -> register_model()
      -> run_segments()
      -> validate_masks()

CritterFrame chooses and materializes training data, records trained-model
provenance, and uses and evaluates the result; the framework-specific training
loop stays external. What this file checks is that the seams hold -- that the
ids a split produced are the ids an export wrote, that the digest an export
recorded is the digest a registration points at, and that a model registered
this way behaves like any other model in a run.

The step that stands in for training is `ThresholdModel`, because what is being
tested is the plumbing on either side of the loop rather than the loop.

Harvested from `scripts/simple_tests/training/training_test.py`.
"""

import json

import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8
PROPORTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def dataset_record(directory):
    with (directory / "dataset.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def train_a_checkpoint(project_path, contents=b"weights, revision 1"):
    """Stand-in for whatever a framework leaves behind."""
    from critterframe.project import paths

    path = paths.models_dir(project_path) / "blobnet.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def test_the_whole_path_holds_together(segmented_project, tmp_path):
    """
    One test walking the workflow end to end, because the value is in the seams
    -- each step is covered on its own elsewhere.
    """
    splits = cf.split_ids(segmented_project, proportions=PROPORTIONS,
                          stratify_by="species", group_by="device", seed=123)

    dataset = tmp_path / "dataset"
    manifest = cf.export_training_data(segmented_project, dataset,
                                       splits=splits, masks=True,
                                       metadata=["species"])
    assert len(manifest) == SPECIMENS
    assert set(manifest["split"]) <= set(splits)

    checkpoint = train_a_checkpoint(segmented_project)
    model = cf.register_model(segmented_project, "blobnet_v1", path=checkpoint,
                              task="segment", framework="torch",
                              training_data=dataset,
                              parameters={"epochs": 40})

    result = cf.run_segments(segmented_project, run_name="custom",
                             steps=[cf.segment(model.attach(ThresholdModel(cutoff=120)))],
                             visualize=False)["organism"]
    assert result["processed"] == SPECIMENS

    masks = mask_records.load_masks(segmented_project)
    assert len(masks) == SPECIMENS


def test_the_split_that_was_exported_is_the_split_that_was_registered(
        segmented_project, tmp_path):
    """
    The join that makes the provenance mean anything: the digest in
    `dataset.json` is the digest the model's record points at, so "this model
    saw these occurrences" is checkable rather than remembered.
    """
    splits = cf.split_ids(segmented_project, proportions=PROPORTIONS, seed=1)
    dataset = tmp_path / "dataset"
    cf.export_training_data(segmented_project, dataset, splits=splits)

    model = cf.register_model(segmented_project, "blobnet_v1",
                              path=train_a_checkpoint(segmented_project),
                              training_data=dataset)

    recorded = model.record["training_data"]["dataset"]
    assert recorded["data_hash"] == dataset_record(dataset)["data_hash"]
    assert recorded["splits"]["train"]["ids_hash"] == \
        dataset_record(dataset)["splits"]["train"]["ids_hash"]


def test_a_frozen_split_survives_the_script_that_made_it(segmented_project,
                                                          tmp_path):
    """
    `split_ids` decides and `define_subset` freezes. Exporting from the subset
    then gives the same dataset as exporting from the ids -- which is what
    makes the two interchangeable in `splits=`.
    """
    splits = cf.split_ids(segmented_project, proportions={"train": 0.75,
                                                          "test": 0.25}, seed=5)
    for name, ids in splits.items():
        cf.define_subset(segmented_project, name, occurrence_ids=ids)

    from_ids = tmp_path / "from_ids"
    from_subsets = tmp_path / "from_subsets"
    cf.export_training_data(segmented_project, from_ids, splits=splits)
    cf.export_training_data(segmented_project, from_subsets,
                            splits={"train": "train", "test": "test"})

    assert dataset_record(from_ids)["data_hash"] == \
        dataset_record(from_subsets)["data_hash"]


def test_training_on_reference_masks_rather_than_the_model_s_own(
        segmented_project, tmp_path):
    """
    Usually the whole point: the reason to train a model is that the automated
    masks weren't good enough, so training on them would teach the new model to
    reproduce the old one's mistakes.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=3))],
                    reference=True, limit=4, visualize=False)

    dataset = tmp_path / "reference_dataset"
    manifest = cf.export_training_data(segmented_project, dataset,
                                       reference=True, masks=True)

    assert len(manifest) == 4
    assert dataset_record(dataset)["reference"] is True


def test_retraining_into_the_same_file_redoes_the_work_below_it(
        segmented_project, tmp_path):
    """
    The failure the fingerprint exists for, at full scale: same registered
    name, same path, same class -- new weights, so the masks and every metric
    measured from them are correctly recomputed rather than silently kept.
    """
    checkpoint = train_a_checkpoint(segmented_project)
    first = cf.register_model(segmented_project, "blobnet_v1", path=checkpoint)

    def run(model):
        return cf.run_segments(
            segmented_project, run_name="custom",
            steps=[cf.segment(model.attach(ThresholdModel(cutoff=120)))],
            visualize=False)["organism"]

    assert run(first)["processed"] == SPECIMENS
    assert run(first)["skipped"] == SPECIMENS

    checkpoint.write_bytes(b"weights, revision 2 -- trained for longer")
    second = cf.register_model(segmented_project, "blobnet_v1", path=checkpoint)
    assert run(second)["processed"] == SPECIMENS


def test_a_registered_model_and_a_bare_one_are_not_the_same_work(
        segmented_project):
    """
    Registering is not cosmetic: it replaces "some ThresholdModel" with "these
    exact weights" in the recipe hash, which is the difference between a model
    that can be traced and one that cannot.
    """
    model = cf.register_model(segmented_project, "blobnet_v1",
                              path=train_a_checkpoint(segmented_project))

    bare = cf.Recipe("segment", "custom", [cf.segment(ThresholdModel())])
    registered = cf.Recipe("segment", "custom",
                           [cf.segment(model.attach(ThresholdModel()))])
    assert bare.hash != registered.hash


def test_the_registry_survives_a_new_session(segmented_project):
    """
    Provenance that only existed in the process that wrote it would be no
    provenance at all.
    """
    cf.register_model(segmented_project, "blobnet_v1",
                      path=train_a_checkpoint(segmented_project),
                      task="segment", base_model="sam2_hiera_large")

    loaded = cf.load_model(segmented_project, "blobnet_v1")
    assert loaded.record["base_model"] == "sam2_hiera_large"
    assert cf.list_models(segmented_project)["blobnet_v1"]["task"] == "segment"


def test_the_model_s_masks_can_be_validated_against_the_reference_set(
        segmented_project, tmp_path):
    """
    The last step of the loop, and the reason the reference table exists: the
    trained model's masks are compared against the ones a person vetted, and
    nothing about that comparison is persisted.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=3))],
                    reference=True, visualize=False)

    model = cf.register_model(segmented_project, "blobnet_v1",
                              path=train_a_checkpoint(segmented_project))
    cf.run_segments(segmented_project, run_name="custom",
                    steps=[cf.segment(model.attach(ThresholdModel(cutoff=120)))],
                    visualize=False)

    scores = cf.validate_masks(segmented_project, visualize=False)
    assert len(scores) == SPECIMENS
    assert scores["iou"].between(0, 1).all()
