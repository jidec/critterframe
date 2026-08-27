"""
Materializing a project as files a trainer can read.

Training data isn't something to assemble separately -- it is a view over what
the project already holds. What this module adds on top is arrangement: which
occurrences, in which split, under which class folder, with a manifest that
names every file and a `dataset.json` recording what the export IS.

Two rules with teeth:

  **splitting and exporting stay separate.** Nothing here chooses proportions.
  A split is a scientific decision that wants to be made once, reviewed, and
  reused; folding it into the exporter would mean re-deciding it every time
  anything is materialized, and two datasets from one project could then
  disagree about what "test" means.

  **an occurrence in two splits raises.** That is the leakage every other
  guarantee in the package exists to prevent, and a written dataset is exactly
  where it would stop being visible.
"""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.training.datasets import (
    _class_folder,
    _relative,
    iterate_segments,
    write_dataset,
)

pytestmark = pytest.mark.slow

SPECIMENS = 8


def splits_of(project_path):
    """A two-way split of the template project's eight specimens."""
    ids = [f"specimen{index}" for index in range(SPECIMENS)]
    return {"train": ids[:6], "val": ids[6:]}


def manifest_of(directory):
    return pd.read_csv(directory / "manifest.csv")


def dataset_record(directory):
    with (directory / "dataset.json").open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The helpers, with no project
# ---------------------------------------------------------------------------


def test_a_class_name_becomes_a_usable_folder_name():
    assert _class_folder({}, "Anax junius") == "Anax_junius"


def test_a_class_holding_a_separator_never_becomes_a_directory():
    """
    "Aeshnidae/Anax" would otherwise quietly nest, and an ImageFolder loader
    would then report a class nobody wrote.
    """
    folder = _class_folder({}, "Aeshnidae/Anax")
    assert "/" not in folder and "\\" not in folder


def test_two_classes_that_would_share_a_folder_raise():
    """
    Merging two classes into one folder would train a model on a taxonomy
    nobody wrote, invisibly.
    """
    folders = {}
    _class_folder(folders, "Anax junius")
    with pytest.raises(ValueError, match="both become folder"):
        _class_folder(folders, "Anax/junius")


def test_a_manifest_path_is_relative_and_posix(tmp_path):
    """
    Relative so the whole directory can be moved to wherever training happens,
    and posix-separated because that move is very often Windows to a Linux
    cluster -- where a backslash is a legal filename character rather than a
    separator.
    """
    written = tmp_path / "train" / "images" / "specimen0.png"
    assert _relative(written, tmp_path) == "train/images/specimen0.png"


# ---------------------------------------------------------------------------
# iterate_segments
# ---------------------------------------------------------------------------


def test_iteration_yields_a_segment_per_masked_occurrence(segmented_project):
    yielded = list(iterate_segments(segmented_project))
    assert len(yielded) == SPECIMENS
    assert all(segment.mask is not None for _id, segment in yielded)


def test_transforms_are_applied_on_the_way_out(segmented_project):
    """
    And using the same chain here as the model will get at inference time is
    the point -- a model trained on background-removed, oriented segments needs
    those operations in front of it when it runs.
    """
    plain = dict(iterate_segments(segmented_project))
    cropped = dict(iterate_segments(segmented_project,
                                    transforms=[cf.crop_to_mask()]))
    assert cropped["specimen0"].shape != plain["specimen0"].shape


def test_unmasked_occurrences_are_skipped_by_default(image_project):
    assert list(iterate_segments(image_project)) == []


def test_they_can_be_included_for_training_on_whole_images(image_project):
    """
    A classifier over photographs the project never segmented: an unsegmented
    occurrence is still training data.
    """
    yielded = list(iterate_segments(image_project, require_mask=False))
    assert len(yielded) == SPECIMENS
    assert all(segment.mask is None for _id, segment in yielded)


# ---------------------------------------------------------------------------
# export_training_data
# ---------------------------------------------------------------------------


def test_an_unsplit_export_is_one_flat_directory(segmented_project, tmp_path):
    out = tmp_path / "flat"
    manifest = cf.export_training_data(segmented_project, out)

    assert len(manifest) == SPECIMENS
    assert (out / "images" / "specimen0.png").exists()
    assert "split" not in manifest.columns


def test_splits_become_directories(segmented_project, tmp_path):
    out = tmp_path / "split"
    manifest = cf.export_training_data(segmented_project, out,
                                       splits=splits_of(segmented_project))

    assert (out / "train" / "images").is_dir()
    assert (out / "val" / "images").is_dir()
    assert manifest["split"].value_counts().to_dict() == {"train": 6, "val": 2}


def test_a_split_may_be_named_as_a_subset_instead_of_listed(segmented_project,
                                                            tmp_path):
    """
    Both are how a project legitimately holds a selection: split_ids hands back
    ids, and a selection worth keeping gets frozen as a subset.
    """
    cf.define_subset(segmented_project, "boxA", column="device", values=["boxA"])
    cf.define_subset(segmented_project, "boxB", column="device", values=["boxB"])

    manifest = cf.export_training_data(segmented_project, tmp_path / "subsets",
                                       splits={"train": "boxA", "val": "boxB"})
    assert manifest["split"].value_counts().to_dict() == {"train": 4, "val": 4}


def test_an_occurrence_in_two_splits_is_refused(segmented_project, tmp_path):
    splits = splits_of(segmented_project)
    splits["val"] = splits["val"] + splits["train"][:1]

    with pytest.raises(ValueError, match="two splits"):
        cf.export_training_data(segmented_project, tmp_path / "leaky",
                                splits=splits)


def test_splits_and_a_limit_are_two_ways_of_saying_which(segmented_project,
                                                          tmp_path):
    with pytest.raises(ValueError, match="not both"):
        cf.export_training_data(segmented_project, tmp_path / "x",
                                splits=splits_of(segmented_project), limit=3)


def test_class_folders_make_an_imagefolder_tree(segmented_project, tmp_path):
    out = tmp_path / "classes"
    manifest = cf.export_training_data(segmented_project, out,
                                       splits=splits_of(segmented_project),
                                       class_by="species")

    assert (out / "train" / "Anax_junius").is_dir()
    assert set(manifest["class"]) == {"Anax junius", "Libellula lydia"}
    assert manifest["image_path"].str.startswith(("train/", "val/")).all()


def test_an_occurrence_with_no_class_is_left_out(segmented_project, tmp_path,
                                                  caplog):
    """
    The class IS the training target here, so an image without one has nothing
    to teach -- unlike a split, where an unlabelled image is still data.
    """
    table = pd.read_parquet(segmented_project / "occurrences.parquet")
    table.loc[:2, "species"] = None
    from critterframe.records.occurrences import save_occurrences
    save_occurrences(segmented_project, table)

    with caplog.at_level("INFO"):
        manifest = cf.export_training_data(segmented_project, tmp_path / "some",
                                           class_by="species")
    assert len(manifest) == SPECIMENS - 3
    assert "no value in 'species'" in caplog.text


def test_masks_are_written_for_segmentation_training(segmented_project, tmp_path):
    """
    0/255 PNGs -- lossless, so a mask boundary survives the round trip exactly.
    """
    out = tmp_path / "seg"
    manifest = cf.export_training_data(segmented_project, out, masks=True)

    mask_file = out / manifest["mask_path"].iloc[0]
    stored = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(stored)) <= {0, 255}
    assert manifest["mask_area"].gt(0).all()


def test_masks_are_off_by_default(segmented_project, tmp_path):
    """What segmentation training needs and what a classifier has no use for."""
    manifest = cf.export_training_data(segmented_project, tmp_path / "images_only")
    assert "mask_path" not in manifest.columns


def test_reference_masks_can_be_exported_instead(segmented_project, tmp_path):
    """
    Usually what you want: the reason to train a model is that the automated
    masks weren't good enough, so training on them would teach the new model to
    reproduce the old one's mistakes.
    """
    from helpers.models import ThresholdModel

    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=3))],
                    reference=True, limit=4, visualize=False)

    manifest = cf.export_training_data(segmented_project, tmp_path / "ref",
                                       reference=True, masks=True)
    assert len(manifest) == 4


def test_metadata_and_metrics_ride_along_in_the_manifest(measured_project,
                                                          tmp_path):
    """
    So a stored trait or QC score can be a training target, a stratification
    key, or a filter downstream without a second join.
    """
    manifest = cf.export_training_data(measured_project, tmp_path / "rich",
                                       metadata=["species", "device"],
                                       metrics=["traits"])
    assert {"species", "device"} <= set(manifest.columns)
    assert "traits__organism__body_length" in manifest.columns


def test_the_dataset_record_says_what_was_exported(segmented_project, tmp_path):
    out = tmp_path / "recorded"
    cf.export_training_data(segmented_project, out,
                            splits=splits_of(segmented_project),
                            transforms=[cf.remove_background()], masks=True)
    record = dataset_record(out)

    assert record["part"] == "organism"
    assert record["masks"] is True
    assert record["splits"]["train"]["count"] == 6
    assert [operation["name"] for operation in record["transforms"]] == [
        "remove_background"]


def test_the_record_identifies_the_data_without_listing_it(segmented_project,
                                                            tmp_path):
    """
    Digests rather than ids: the record must be able to prove which set was
    used, which a digest does, and thousands of ids would make it unreadable.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    splits = splits_of(segmented_project)
    cf.export_training_data(segmented_project, first, splits=splits)
    cf.export_training_data(segmented_project, second, splits=splits)

    assert dataset_record(first)["data_hash"] == dataset_record(second)["data_hash"]
    assert dataset_record(first)["splits"]["train"]["ids_hash"] == \
        dataset_record(second)["splits"]["train"]["ids_hash"]


def test_a_different_selection_is_a_different_dataset(segmented_project, tmp_path):
    splits = splits_of(segmented_project)
    other = {"train": splits["train"][:4], "val": splits["val"]}

    cf.export_training_data(segmented_project, tmp_path / "a", splits=splits)
    cf.export_training_data(segmented_project, tmp_path / "b", splits=other)
    assert dataset_record(tmp_path / "a")["data_hash"] != \
        dataset_record(tmp_path / "b")["data_hash"]


def test_a_different_transform_chain_is_a_different_dataset(segmented_project,
                                                             tmp_path):
    """
    Because it changes what the model will see -- which is exactly the thing a
    registered model's provenance needs to be able to distinguish.
    """
    cf.export_training_data(segmented_project, tmp_path / "a")
    cf.export_training_data(segmented_project, tmp_path / "b",
                            transforms=[cf.remove_background()])
    assert dataset_record(tmp_path / "a")["data_hash"] != \
        dataset_record(tmp_path / "b")["data_hash"]


def test_the_record_describes_what_was_written_not_what_was_asked_for(
        image_project, tmp_path):
    """
    Occurrences drop out for want of a mask or a class, and a record that
    counted the request would overstate the training set.
    """
    from helpers.models import ThresholdModel

    cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                    limit=3, visualize=False)
    out = tmp_path / "partial"
    cf.export_training_data(image_project, out,
                            splits={"train": [f"specimen{i}" for i in range(8)]})

    assert dataset_record(out)["splits"]["train"]["count"] == 3


def test_exporting_from_an_unsegmented_project_says_so(image_project, tmp_path,
                                                       caplog):
    with caplog.at_level("WARNING"):
        manifest = cf.export_training_data(image_project, tmp_path / "empty")
    assert manifest.empty
    assert "nothing exported" in caplog.text


# ---------------------------------------------------------------------------
# write_dataset -- the flat form the extensions are written against
# ---------------------------------------------------------------------------


def test_write_dataset_keeps_its_own_shape(segmented_project, tmp_path):
    out = tmp_path / "legacy"
    manifest = write_dataset(segmented_project, out, label_columns=["species"])

    assert (out / "images").is_dir() and (out / "masks").is_dir()
    assert {"occurrence_id", "part", "image_path", "mask_path", "height",
            "width", "mask_area", "species"} <= set(manifest.columns)
