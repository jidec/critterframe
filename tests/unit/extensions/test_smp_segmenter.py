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
    inat_insects.metrics.bioencoder.BioEncoderModel.
    """
    first = segmentation.smp_segmenter(checkpoint="a.pt").identity()
    second = segmentation.smp_segmenter(checkpoint="b.pt").identity()
    assert first != second
    assert first["checkpoint"] == "a.pt"


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
                                size=64, num_epochs=1, batch_size=2, num_workers=0)
    assert checkpoint.exists()

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
