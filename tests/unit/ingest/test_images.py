"""
Ingesting a folder of images: bytes into the store untouched, one occurrence
row per file, and the folder as the source of truth.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records.occurrences import ID_COL
from critterframe.storage.imagestore import ImageStore
from helpers.synthetic import draw_specimen
from helpers.imports import imports_of


# ---------------------------------------------------------------------------
# ingest_images
# ---------------------------------------------------------------------------


def test_images_become_occurrences_keyed_by_their_stems(tmp_path, image_dir):
    project = tmp_path / "project"
    summary = cf.ingest_images(project, image_dir)

    assert summary == {
        "attempted": 3,
        "processed": 3,
        "skipped": 0,
        "no_input": 0,
        "failed": 0,
        "failures": [],
        "flags": {},
        "occurrences": 3,
    }
    table = pd.read_parquet(paths.occurrences_path(project))
    assert sorted(table[ID_COL]) == ["spec0", "spec1", "spec2"]


def test_the_stored_bytes_are_the_file_s_own(tmp_path, image_dir):
    """
    Copied byte for byte in whatever format they already are. The decode that
    happens on the way is only to check readability and measure dimensions --
    storing that array instead would flatten every image to 8-bit BGR.
    """
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)

    original = (image_dir / "spec0.png").read_bytes()
    with ImageStore(project, readonly=True) as store:
        assert store.get_bytes("spec0") == original


def test_dimensions_are_recorded_from_the_file(tmp_path, image_dir):
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)

    table = pd.read_parquet(paths.occurrences_path(project)).set_index(ID_COL)
    assert table.loc["spec0", "image_width"] == 280
    assert table.loc["spec0", "image_height"] == 220
    assert table.loc["spec0", "source_format"] == "png"


def test_no_image_url_is_invented(tmp_path, image_dir):
    """URLs exist so download_images() can fetch pixels; these are already here."""
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)
    assert "image_url" not in pd.read_parquet(paths.occurrences_path(project))


def test_a_manifest_is_archived_rather_than_the_pixels(tmp_path, image_dir):
    """
    Copying every image into imports/ would double a project's largest storage
    cost to duplicate files already on disk. What you would actually consult
    later is where each image came from and whether it has changed.
    """
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)

    archived = imports_of(project)
    assert len(archived) == 1
    assert archived[0].name.startswith("images_images_")

    manifest = pd.read_csv(archived[0])
    assert {"occurrence_id", "source_path", "source_bytes", "source_mtime"} <= set(manifest.columns)
    assert not list(paths.raw_imports_dir(project).glob("*.png"))
    assert not list(paths.raw_imports_dir(project).glob(".manifest.csv"))


def test_an_unchanged_folder_is_not_archived_twice(tmp_path, image_dir):
    """
    The same bytes are never stored twice in raw_imports/, the rule occurrence
    imports already follow. Every image is still re-read; only the manifest,
    which says the folder is exactly as it was, is recognized.
    """
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)
    cf.ingest_images(project, image_dir)

    assert len(imports_of(project)) == 1


def test_a_changed_folder_archives_a_new_manifest(tmp_path, image_dir):
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)
    cv2.imwrite(str(image_dir / "spec0.png"), np.full((50, 60, 3), 200, np.uint8))
    cf.ingest_images(project, image_dir)

    assert len(imports_of(project)) == 2


def test_metadata_is_joined_onto_the_occurrences(tmp_path, image_dir):
    """
    How a metadata CSV and a folder of images become ONE snapshot -- ingesting
    them separately would have the second replace the first.
    """
    project = tmp_path / "project"
    metadata = pd.DataFrame(
        {"occurrence_id": ["spec0", "spec1", "spec2"], "species": ["Anax", "Anax", "Libellula"]}
    )
    cf.ingest_images(project, image_dir, metadata=metadata)

    table = pd.read_parquet(paths.occurrences_path(project)).set_index(ID_COL)
    assert table.loc["spec2", "species"] == "Libellula"


def test_metadata_for_an_absent_image_is_left_behind(tmp_path, image_dir):
    """
    The FOLDER is the source of truth. A metadata row for a file that isn't
    there describes nothing this project can process.
    """
    project = tmp_path / "project"
    metadata = pd.DataFrame({"occurrence_id": ["spec0", "ghost"], "species": ["Anax", "Libellula"]})
    cf.ingest_images(project, image_dir, metadata=metadata)

    table = pd.read_parquet(paths.occurrences_path(project))
    assert "ghost" not in table[ID_COL].tolist()


def test_colliding_stems_are_refused_before_anything_is_written(tmp_path, image_dir):
    """
    photo.jpg and photo.tiff would share an occurrence id. Reported as the
    colliding PATHS, because the downstream duplicate-id error names the id --
    which doesn't tell you which two files to rename.
    """
    cv2.imwrite(str(image_dir / "spec0.jpg"), draw_specimen(0))

    project = tmp_path / "project"
    with pytest.raises(ValueError, match="filename stem"):
        cf.ingest_images(project, image_dir)
    assert not paths.occurrences_path(project).exists()


def test_an_unreadable_file_is_counted_not_fatal(tmp_path, image_dir):
    (image_dir / "broken.png").write_bytes(b"not an image")

    summary = cf.ingest_images(tmp_path / "project", image_dir)
    assert (summary["processed"], summary["failed"]) == (3, 1)
    assert summary["failures"][0]["path"].endswith("broken.png")


def test_batching_still_saves_the_remainder(tmp_path, image_dir):
    """
    The last partial batch has to be flushed after the loop, or the tail of
    every ingest would be silently lost.
    """
    project = tmp_path / "project"
    summary = cf.ingest_images(project, image_dir, batch_size=2)

    assert summary["processed"] == 3
    with ImageStore(project, readonly=True) as store:
        assert sorted(store.keys()) == ["spec0", "spec1", "spec2"]


def test_an_empty_folder_leaves_no_project_behind(tmp_path):
    """
    Returning early without writing an occurrence table is the honest outcome:
    the project stays invalid rather than claiming zero occurrences.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    project = tmp_path / "project"

    summary = cf.ingest_images(project, empty)
    assert summary["processed"] == 0
    assert not paths.occurrences_path(project).exists()


def test_only_matching_patterns_are_ingested(tmp_path, image_dir):
    (image_dir / "notes.txt").write_text("not an image")
    summary = cf.ingest_images(tmp_path / "project", image_dir)
    assert summary["attempted"] == 3


def test_subdirectories_are_searched_only_when_asked(tmp_path, image_dir):
    nested = image_dir / "more"
    nested.mkdir()
    cv2.imwrite(str(nested / "spec9.png"), draw_specimen(4))

    assert cf.ingest_images(tmp_path / "flat", image_dir)["processed"] == 3
    assert cf.ingest_images(tmp_path / "deep", image_dir, recursive=True)["processed"] == 4


def test_ids_only_ever_come_from_filenames(tmp_path, image_dir):
    """
    The flag exists to make that explicit: filenames have to be unique and
    stable, since renaming one later orphans every mask and metric keyed to it.
    """
    with pytest.raises(ValueError, match="ids come from filenames"):
        cf.ingest_images(tmp_path / "project", image_dir, id_from_stem=False)


def test_reingesting_follows_the_folder(tmp_path, image_dir):
    """
    Every matched file is re-read and re-stored, which is what makes replacing
    a file with a corrected version work -- the store follows the folder rather
    than keeping the first version it saw.
    """
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)

    corrected = np.full((50, 60, 3), 200, np.uint8)
    cv2.imwrite(str(image_dir / "spec0.png"), corrected)
    cf.ingest_images(project, image_dir)

    with ImageStore(project, readonly=True) as store:
        assert store.get("spec0").shape == (50, 60, 3)


def test_a_removed_file_loses_its_occurrence_row(tmp_path, image_dir):
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)
    (image_dir / "spec2.png").unlink()
    cf.ingest_images(project, image_dir)

    table = pd.read_parquet(paths.occurrences_path(project))
    assert sorted(table[ID_COL]) == ["spec0", "spec1"]

    # Its image is still in the store, keyed by an id nothing now references.
    with ImageStore(project, readonly=True) as store:
        assert store.has("spec2")
