"""
The trainable UNet++ segmenter: identity without torch, dataset export against
a real project, and one real train+load+predict round trip.

Mirrors tests/unit/segmentation/test_groundedsam.py's split: everything that
doesn't need real weights is tested unconditionally; actually running the
network is `gpu`-marked and deselected by default.
"""

import sys

import numpy as np
import pytest

from critterframe.extensions.smp_segmenter import segmentation
from critterframe.recipes import Recipe
from critterframe.segmentation.run import segment


# ---------------------------------------------------------------------------
# Construction and identity, without torch
# ---------------------------------------------------------------------------


def test_constructing_the_model_loads_no_weights():
    """__init__ only stores configuration, same discipline as GroundedSAM2."""
    model = segmentation.smp_segmenter()
    assert model.model is None


def test_constructing_the_model_imports_no_torch():
    for name in ("torch", "segmentation_models_pytorch"):
        sys.modules.pop(name, None)

    segmentation.smp_segmenter()
    assert "torch" not in sys.modules
    assert "segmentation_models_pytorch" not in sys.modules


def test_a_recipe_naming_it_can_be_hashed_without_torch():
    recipe = Recipe("segment", "aux", [segment(segmentation.smp_segmenter())],
                    part="organism")
    assert len(recipe.hash) == 16


def test_the_checkpoint_is_part_of_the_identity():
    """
    Two checkpoints are not equivalent work, same reasoning as
    bioencoder.embedding.BioEncoderModel.
    """
    first = segmentation.smp_segmenter(checkpoint="a.pt").identity()
    second = segmentation.smp_segmenter(checkpoint="b.pt").identity()
    assert first != second
    assert first["checkpoint"] == "a.pt"


def test_the_identity_is_the_checkpoints_content_not_its_path(tmp_path):
    """
    CLAUDE.md's registered-model rule, applied to a standalone segmenter too:
    two copies of one checkpoint are the same model, and two different
    checkpoints written to one path over a project's life are not. A bare path
    got both backwards.
    """
    first = tmp_path / "a.pt"
    first.write_bytes(b"weights-one")
    copied = tmp_path / "copy_of_a.pt"
    copied.write_bytes(b"weights-one")
    retrained = tmp_path / "retrained.pt"
    retrained.write_bytes(b"weights-two")

    identity = segmentation.smp_segmenter(checkpoint=first).identity()
    assert identity == segmentation.smp_segmenter(checkpoint=copied).identity()
    assert identity != segmentation.smp_segmenter(checkpoint=retrained).identity()
    assert str(first) not in str(identity)


def test_the_checkpoint_is_read_once(tmp_path, monkeypatch):
    """
    Recipe.hash asks for identity() many times over a run, and a checkpoint is
    hundreds of megabytes -- re-hashing it per call would cost more than the
    segmentation.
    """
    from critterframe.records import models as model_records

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")

    reads = []
    real = model_records.fingerprint_file
    monkeypatch.setattr(segmentation, "fingerprint_file",
                        lambda path: reads.append(path) or real(path))

    model = segmentation.smp_segmenter(checkpoint=checkpoint)
    assert model.identity() == model.identity()
    assert len(reads) == 1


def test_the_encoder_and_size_are_part_of_the_identity():
    base = segmentation.smp_segmenter().identity()
    assert segmentation.smp_segmenter(encoder_name="resnet18").identity() != base
    assert segmentation.smp_segmenter(size=512).identity() != base


def test_the_device_is_not_part_of_the_identity():
    """Running the same weights on CPU and GPU is the same work."""
    assert (segmentation.smp_segmenter(device="cpu").identity()
            == segmentation.smp_segmenter(device="cuda").identity())


def test_no_checkpoint_reads_as_untrained_in_the_identity():
    assert segmentation.smp_segmenter().identity()["checkpoint"] is None


# ---------------------------------------------------------------------------
# prepare_dataset(): real, and torch-free
# ---------------------------------------------------------------------------


def test_prepare_dataset_writes_a_split_image_and_mask_dataset(segmented_project, tmp_path):
    """
    No torch import anywhere in this path -- prepare_dataset() is a thin
    wrapper over core split_ids()/export_training_data(), so it works without
    the [torch] extra installed, same as the rest of dataset prep in this
    package.
    """
    assert "torch" not in sys.modules

    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    manifest = training.prepare_dataset(
        segmented_project, dataset_dir, reference=False,
        fractions={"train": 0.75, "val": 0.25})

    assert set(manifest["split"].unique()) <= {"train", "val"}
    assert (dataset_dir / "dataset.json").exists()
    assert (dataset_dir / "manifest.csv").exists()
    for _, row in manifest.iterrows():
        assert (dataset_dir / row["image_path"]).exists()
        assert (dataset_dir / row["mask_path"]).exists()


def test_prepare_dataset_defaults_to_reference_masks(segmented_project, tmp_path):
    """
    Training on canonical masks would teach this model the bundled segmenter's
    own mistakes -- the same reasoning training.datasets.iterate_segments
    documents for reference=True.
    """
    from critterframe.extensions.smp_segmenter import training

    # segmented_project has canonical masks only, no reference masks, so the
    # reference=True default should export nothing rather than silently fall
    # back to canonical.
    manifest = training.prepare_dataset(segmented_project, tmp_path / "dataset")
    assert manifest.empty


# ---------------------------------------------------------------------------
# register_trained(): what load_registered() needs, recorded
# ---------------------------------------------------------------------------


def test_register_trained_records_what_loading_it_back_needs(segmented_project,
                                                             tmp_path):
    """
    load_registered() reads encoder_name/size out of the registry's own
    parameters. Registered by hand without them -- which is what this module's
    own example used to show -- a model loads under whatever the module
    defaults happen to be at load time, silently, which is the exact failure
    load_registered() exists to prevent.
    """
    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    training.prepare_dataset(segmented_project, dataset_dir, reference=False,
                             fractions={"train": 1.0})
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not really weights, but bytes to fingerprint")

    registered = training.register_trained(
        segmented_project, "aux_v1", checkpoint, dataset_dir,
        encoder_name="resnet18", size=256)

    assert registered.record["parameters"] == {"encoder_name": "resnet18",
                                               "size": 256}
    assert registered.record["task"] == "segment"
    assert registered.record["fingerprint"]
    # The dataset is recorded by what it holds, not just where it sat.
    assert registered.record["training_data"]["dataset"]["data_hash"]


def test_a_registered_model_loads_back_with_the_architecture_it_was_trained_with(
        segmented_project, tmp_path):
    """The round trip: what register_trained wrote is what load_registered builds."""
    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    training.prepare_dataset(segmented_project, dataset_dir, reference=False,
                             fractions={"train": 1.0})
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"bytes")
    training.register_trained(segmented_project, "aux_v1", checkpoint,
                              dataset_dir, encoder_name="resnet18", size=256)

    loaded = segmentation.load_registered(segmented_project, "aux_v1")
    assert (loaded.runtime.encoder_name, loaded.runtime.size) == ("resnet18", 256)
    # identity() still answers from the registry, not from the attached
    # segmenter's own fingerprint -- a registered model's provenance wins.
    assert loaded.identity()["class"] == "RegisteredModel"


def test_training_curves_are_drawn_without_torch(tmp_path):
    """The figures train() ends with need matplotlib only, so they're checked unconditionally."""
    from critterframe.extensions.smp_segmenter import training
    from critterframe.visualization.pipeline import open_report

    history = {f"{phase}_{metric}": [0.9, 0.5, 0.4] for phase in ("train", "val")
               for metric in ("loss", "f1", "auroc", "iou")}
    report = open_report(tmp_path, "train__smp_resnet18", "abc").begin([])
    training._training_figures(report, history, ("train", "val"), best_epoch=3)

    from critterframe.project import paths
    written = sorted(path.name for path in paths.pipeline_dir(tmp_path).glob("*.png"))
    assert written == ["train__smp_resnet18_abc__loss.png", "train__smp_resnet18_abc__scores.png"]
    assert "torch" not in sys.modules


# ---------------------------------------------------------------------------
# Actually training and running the network
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_a_trained_checkpoint_loads_and_predicts(segmented_project, tmp_path,
                                                  draw_specimen):
    """
    Deselected by default. One epoch over a tiny synthetic dataset, on a small
    encoder -- the point is proving the plumbing (export -> train -> save ->
    load -> predict) works end to end, not that resnet18 segments well.
    """
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    manifest = training.prepare_dataset(
        segmented_project, dataset_dir, reference=False,
        fractions={"train": 0.75, "val": 0.25})

    checkpoint = training.train(manifest, dataset_dir, encoder_name="resnet18",
                                size=64, num_epochs=1, batch_size=2, num_workers=0,
                                project_path=segmented_project)
    assert checkpoint.exists()

    from critterframe.project import paths
    written = sorted(path.name for path in
                     paths.pipeline_dir(segmented_project).glob("train__smp_resnet18_*"))
    assert any(name.endswith("__epoch0001.jpg") for name in written)
    assert any(name.endswith("__loss.png") for name in written)

    model = segmentation.smp_segmenter(checkpoint=checkpoint,
                                       encoder_name="resnet18", size=64)
    image = draw_specimen(0)
    mask, score, info = model.predict(image[..., ::-1])  # RGB, as segment() passes it

    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool or set(np.unique(mask)) <= {0, 1}
    assert score is None or 0 <= float(score) <= 1
    assert isinstance(info, dict)


@pytest.mark.gpu
def test_training_with_no_val_split_saves_the_final_epoch(segmented_project,
                                                           tmp_path, caplog):
    """
    val_split=None: a reference set too small to spare a third split still
    trains, with no early stopping -- the tradeoff logged rather than hidden.
    """
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    manifest = training.prepare_dataset(
        segmented_project, dataset_dir, reference=False,
        fractions={"train": 0.75, "test": 0.25})
    assert set(manifest["split"].unique()) <= {"train", "test"}

    with caplog.at_level("WARNING"):
        checkpoint = training.train(manifest, dataset_dir, encoder_name="resnet18",
                                    size=64, num_epochs=1, batch_size=2,
                                    num_workers=0, val_split=None)

    assert checkpoint.exists()
    assert "no validation split" in caplog.text


@pytest.mark.gpu
def test_a_manifest_missing_the_named_val_split_still_raises(segmented_project,
                                                              tmp_path):
    """
    val_split=None is the only opt-out -- leaving the default val_split="val"
    against a manifest with no "val" split is still a mistake worth failing
    loudly on, not something to infer. (train() imports torch unconditionally
    before it ever reaches this check, so this needs torch installed same as
    every other train() test, even though it never trains an epoch.)
    """
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")

    from critterframe.extensions.smp_segmenter import training

    dataset_dir = tmp_path / "dataset"
    manifest = training.prepare_dataset(
        segmented_project, dataset_dir, reference=False,
        fractions={"train": 0.75, "test": 0.25})

    with pytest.raises(ValueError, match="no usable rows with split=='val'"):
        training.train(manifest, dataset_dir, num_epochs=1)
