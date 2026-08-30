"""
Getting data in: archive first, parse second, snapshot always.

The ordering is the point. The source file is copied into `imports/` BEFORE it
is read, which is what makes `drop=` safe -- rows the source declared are not
organisms never reach the occurrence table, and every one of them is still in
the archive if that judgement was wrong. Nothing else in the package deletes a
row, and this only gets to because what it excludes is a fact the SOURCE
reported rather than a judgement this project made.

Harvested in part from `scripts/simple_tests/ingest_test.py`, which had three of
these assertions and printed the rest.
"""

import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records.occurrences import ID_COL
from critterframe.storage.imagestore import ImageStore
from helpers.synthetic import draw_specimen


@pytest.fixture
def source_csv(tmp_path):
    """A small occurrence export with a determination column to drop on."""
    path = tmp_path / "export.csv"
    pd.DataFrame({
        "detection_id": [1, 2, 3, 4],
        "photo": [f"http://example/{index}.jpg" for index in range(4)],
        "determination_name": ["Noctuidae", "Not Lepidoptera", "Geometridae",
                               "Debris"],
        "captured": ["2024-05-01", "2024-05-02", "bad date", "2024-05-04"],
    }).to_csv(path, index=False)
    return path


def imports_of(project_path):
    return sorted(paths.imports_dir(project_path).glob("*.csv"))


# ---------------------------------------------------------------------------
# ingest_occurrences
# ---------------------------------------------------------------------------


def test_ingest_creates_the_project_lazily(tmp_path, source_csv):
    """
    The directory doesn't have to exist: a project comes into being as its
    first writer creates what it needs, and this is normally that writer.
    """
    project = tmp_path / "new_project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    assert paths.occurrences_path(project).exists()
    assert not paths.images_path(project).exists()
    assert not paths.masks_path(project).exists()


def test_the_source_is_archived_byte_for_byte(tmp_path, source_csv):
    """
    The recovery path if an ingest was ever wrong -- so it is a copy of the
    file, not a re-serialization of what was parsed out of it.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    archived = imports_of(project)
    assert len(archived) == 1
    assert archived[0].read_bytes() == source_csv.read_bytes()


def test_the_archive_is_dated_and_prefixed(tmp_path, source_csv):
    """
    Globbed rather than named exactly: the date is today's, and a test that
    computed it would be asserting the same expression twice.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                          name_prefix="occurrences_antenna_199")

    name = imports_of(project)[0].stem
    assert name.startswith("occurrences_antenna_199_")
    datetime.date.fromisoformat(name.rsplit("_", 1)[-1])


def test_two_imports_on_one_day_do_not_clobber_each_other(tmp_path, source_csv,
                                                          monkeypatch):
    """
    The one place the clock is frozen rather than stripped, because here the
    date IS the behaviour: the suffix only appears when two imports share a
    day. Computing today's date in the test would flake once a year, at
    midnight, in a way nobody could reproduce.

    The dated name is built by `paths.import_path` (the whole project layout,
    filenames included, lives in one module), and `paths.py` does `from datetime
    import date` -- so that is the module the name to patch lives on.
    """
    class FixedDate:
        @staticmethod
        def today():
            return datetime.date(2026, 3, 14)

    monkeypatch.setattr(paths, "date", FixedDate)
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    assert [path.name for path in imports_of(project)] == [
        "occurrences_2026-03-14.csv", "occurrences_2026-03-14_1.csv"]


def test_ingest_normalizes_and_returns_the_table(tmp_path, source_csv):
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                                  image_url_col="photo",
                                  datetime_cols=["captured"])

    assert table[ID_COL].tolist() == ["1", "2", "3", "4"]
    assert "image_url" in table.columns
    assert pd.isna(table["captured"].iloc[2])


def test_a_transform_can_derive_columns(tmp_path, source_csv):
    """
    How an extension adds what its source needs without this function growing a
    parameter per quirk.
    """
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id",
        transform=lambda df: df.assign(site=df[ID_COL].str[0]))
    assert "site" in table.columns


def test_drop_keeps_declared_non_organisms_out_of_the_table(tmp_path, source_csv):
    """
    The occurrence contract, not a filter: every row in this table asserts one
    focal organism, and a detection the source classified as debris asserts
    nothing.
    """
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id",
        drop={"determination_name": ["Not Lepidoptera", "Debris"]})

    assert table[ID_COL].tolist() == ["1", "3"]
    assert "Not Lepidoptera" not in set(table["determination_name"])


def test_the_dropped_rows_are_still_in_the_archive(tmp_path, source_csv):
    """What makes dropping safe: the import keeps everything the source sent."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                          drop={"determination_name": ["Not Lepidoptera"]})

    assert len(pd.read_csv(imports_of(project)[0])) == 4


def test_drop_is_applied_after_the_transform(tmp_path, source_csv):
    """So a rule may name a column the transform joined in."""
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id",
        transform=lambda df: df.assign(verdict=df["determination_name"]),
        drop={"verdict": ["Debris"]})
    assert "4" not in table[ID_COL].tolist()


def test_a_drop_rule_on_an_unknown_column_raises(tmp_path, source_csv):
    """
    A typo that silently matched nothing would read as "there was none of
    that here", which is the wrong answer to have believed.
    """
    project = tmp_path / "project"
    with pytest.raises(KeyError, match="rule column"):
        cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                              drop={"determinaton_name": ["Debris"]})


def test_reingesting_replaces_rather_than_appends(tmp_path, source_csv):
    """
    The CSV states what the project's occurrences ARE, in full. To add
    occurrences you add rows to the CSV -- not ingest a second file of new ones.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    smaller = tmp_path / "smaller.csv"
    pd.DataFrame({"detection_id": [1]}).to_csv(smaller, index=False)
    table = cf.ingest_occurrences(project, smaller, id_col="detection_id")

    assert table[ID_COL].tolist() == ["1"]
    assert len(imports_of(project)) == 2


def test_a_duplicate_id_stops_the_ingest(tmp_path):
    """
    And stops it BEFORE writing, so the project keeps whatever it had rather
    than being half-replaced.
    """
    source = tmp_path / "dupes.csv"
    pd.DataFrame({"occurrence_id": ["a", "a"]}).to_csv(source, index=False)

    project = tmp_path / "project"
    with pytest.raises(ValueError, match="duplicate occurrence id"):
        cf.ingest_occurrences(project, source)
    assert not paths.occurrences_path(project).exists()


# ---------------------------------------------------------------------------
# ingest_images
# ---------------------------------------------------------------------------


@pytest.fixture
def image_dir(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for index in range(3):
        cv2.imwrite(str(directory / f"spec{index}.png"), draw_specimen(index))
    return directory


def test_images_become_occurrences_keyed_by_their_stems(tmp_path, image_dir):
    project = tmp_path / "project"
    summary = cf.ingest_images(project, image_dir)

    assert summary == {"attempted": 3, "saved": 3, "failed": 0, "failures": [],
                       "occurrences": 3}
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
    assert {"occurrence_id", "source_path", "source_bytes",
            "source_mtime"} <= set(manifest.columns)
    assert not list(paths.imports_dir(project).glob("*.png"))
    assert not list(paths.imports_dir(project).glob(".manifest.csv"))


def test_metadata_is_joined_onto_the_occurrences(tmp_path, image_dir):
    """
    How a metadata CSV and a folder of images become ONE snapshot -- ingesting
    them separately would have the second replace the first.
    """
    project = tmp_path / "project"
    metadata = pd.DataFrame({"occurrence_id": ["spec0", "spec1", "spec2"],
                             "species": ["Anax", "Anax", "Libellula"]})
    cf.ingest_images(project, image_dir, metadata=metadata)

    table = pd.read_parquet(paths.occurrences_path(project)).set_index(ID_COL)
    assert table.loc["spec2", "species"] == "Libellula"


def test_metadata_for_an_absent_image_is_left_behind(tmp_path, image_dir):
    """
    The FOLDER is the source of truth. A metadata row for a file that isn't
    there describes nothing this project can process.
    """
    project = tmp_path / "project"
    metadata = pd.DataFrame({"occurrence_id": ["spec0", "ghost"],
                             "species": ["Anax", "Libellula"]})
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
    assert (summary["saved"], summary["failed"]) == (3, 1)
    assert summary["failures"][0]["path"].endswith("broken.png")


def test_batching_still_saves_the_remainder(tmp_path, image_dir):
    """
    The last partial batch has to be flushed after the loop, or the tail of
    every ingest would be silently lost.
    """
    project = tmp_path / "project"
    summary = cf.ingest_images(project, image_dir, batch_size=2)

    assert summary["saved"] == 3
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
    assert summary["saved"] == 0
    assert not paths.occurrences_path(project).exists()


def test_only_matching_patterns_are_ingested(tmp_path, image_dir):
    (image_dir / "notes.txt").write_text("not an image")
    summary = cf.ingest_images(tmp_path / "project", image_dir)
    assert summary["attempted"] == 3


def test_subdirectories_are_searched_only_when_asked(tmp_path, image_dir):
    nested = image_dir / "more"
    nested.mkdir()
    cv2.imwrite(str(nested / "spec9.png"), draw_specimen(4))

    assert cf.ingest_images(tmp_path / "flat", image_dir)["saved"] == 3
    assert cf.ingest_images(tmp_path / "deep", image_dir,
                            recursive=True)["saved"] == 4


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
