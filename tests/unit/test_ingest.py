"""
Getting data in: archive first, parse second, snapshot always.

The ordering is the point. The source file is copied into `raw_imports/`
BEFORE it is read, which is what makes `drop=` safe -- rows the source
declared are not organisms never reach the occurrence table, and every one of
them is still in the archive if that judgement was wrong. Nothing else in the
package deletes a row, and this only gets to because what it excludes is a
fact the SOURCE reported rather than a judgement this project made.

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
    return sorted(paths.raw_imports_dir(project_path).glob("*.csv"))


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


def test_two_different_imports_on_one_day_do_not_clobber_each_other(
        tmp_path, source_csv, monkeypatch):
    """
    The one place the clock is frozen rather than stripped, because here the
    date IS the behaviour: the suffix only appears when two GENUINELY
    DIFFERENT raw imports share a day. Computing today's date in the test
    would flake once a year, at midnight, in a way nobody could reproduce.

    The dated name is built by `paths.raw_import_path` (the whole project layout,
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

    revised = tmp_path / "revised.csv"
    pd.read_csv(source_csv).assign(detection_id=lambda df: df["detection_id"] + 10) \
        .to_csv(revised, index=False)
    cf.ingest_occurrences(project, revised, id_col="detection_id",
                          name_prefix="occurrences")

    assert [path.name for path in imports_of(project)] == [
        "occurrences_2026-03-14.csv", "occurrences_2026-03-14_1.csv"]


def test_reingesting_the_same_raw_import_the_same_way_is_a_no_op(tmp_path, source_csv):
    """
    The idempotency guarantee: identical raw content plus identical decisions
    is recognized as work already done, so a scheduled re-pull that finds
    nothing new doesn't re-archive, re-parse, or re-save anything.
    """
    project = tmp_path / "project"
    first = cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                                  drop={"determination_name": ["Debris"]})
    second = cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                                   drop={"determination_name": ["Debris"]})

    assert first[ID_COL].tolist() == second[ID_COL].tolist()
    assert len(imports_of(project)) == 1   # no duplicate archived


def test_reingesting_the_same_raw_import_a_different_way_is_not_a_no_op(
        tmp_path, source_csv):
    """
    Different decisions over identical content is a different IMPORT, even
    though it's the same raw import -- the whole point of the two-tier
    vocabulary. The raw content is still deduplicated: one raw file serves
    both.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", drop=None)
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id",
        drop={"determination_name": ["Not Lepidoptera", "Debris"]})

    assert table[ID_COL].tolist() == ["1", "3"]
    assert len(imports_of(project)) == 1   # same raw content, stored once
    assert len(cf.load_imports(project)) == 2   # two distinct imports


def test_already_ingested_answers_from_raw_bytes_alone(tmp_path, source_csv):
    """
    already_ingested exists for a caller whose own parse of raw_bytes is
    itself expensive (see gbif_darwincore_inat.ingest) -- it has to answer
    correctly with no parsed df in hand, and agree exactly with what
    ingest_occurrences(df=...) would decide for the same inputs.
    """
    from critterframe.ingest import already_ingested

    project = tmp_path / "project"
    raw_bytes = source_csv.read_bytes()
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})

    assert already_ingested(project, raw_bytes, **kwargs) is False

    cf.ingest_occurrences(project, source_csv, **kwargs)
    assert already_ingested(project, raw_bytes, **kwargs) is True

    # A different decision over the same bytes is a different import.
    assert already_ingested(project, raw_bytes, id_col="detection_id",
                            drop=None) is False


# ---------------------------------------------------------------------------
# trust_source_file_unchanged
# ---------------------------------------------------------------------------


def test_trust_source_file_unchanged_skips_the_read_on_a_repeat_call(
        tmp_path, source_csv, monkeypatch):
    """
    Same path, size, and mtime as a previous import: the content is never
    read again, only the cheap fingerprint is checked.
    """
    from pathlib import Path

    project = tmp_path / "project"
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})
    first = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True,
                                  **kwargs)

    def _raise(self, *args, **kwargs):
        raise AssertionError("read_bytes should not be called on a fingerprint hit")
    monkeypatch.setattr(Path, "read_bytes", _raise)

    second = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True,
                                   **kwargs)
    assert second[ID_COL].tolist() == first[ID_COL].tolist()
    assert len(imports_of(project)) == 1


def test_trust_source_file_unchanged_falls_back_when_the_file_changed(tmp_path, source_csv):
    """
    A fingerprint miss -- content, and so size and mtime, changed -- still
    does a real read+hash and a correct re-ingest.
    """
    project = tmp_path / "project"
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})
    cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True, **kwargs)

    revised = pd.read_csv(source_csv)
    revised.loc[len(revised)] = [5, "http://example/5.jpg", "Noctuidae", "2024-05-05"]
    revised.to_csv(source_csv, index=False)

    table = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True,
                                  **kwargs)
    assert "5" in table[ID_COL].tolist()
    assert len(cf.load_imports(project)) == 2


def test_trust_source_file_unchanged_still_reingests_on_a_changed_decision(
        tmp_path, source_csv):
    """
    The file's fingerprint matches, but drop= changed -- the borrowed
    raw_hash produces a different import_hash, so this correctly does NOT
    skip, unlike a naive "same file, always skip" shortcut would.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                          trust_source_file_unchanged=True, drop=None)
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", trust_source_file_unchanged=True,
        drop={"determination_name": ["Not Lepidoptera", "Debris"]})

    assert table[ID_COL].tolist() == ["1", "3"]
    assert len(cf.load_imports(project)) == 2


def test_trust_source_file_unchanged_defaults_off(tmp_path, source_csv, monkeypatch):
    """The fingerprint shortcut only applies when explicitly asked for."""
    from pathlib import Path

    project = tmp_path / "project"
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})
    cf.ingest_occurrences(project, source_csv, **kwargs)

    original_read_bytes = Path.read_bytes
    calls = []

    def _spy(self, *args, **kwargs):
        calls.append(self)
        return original_read_bytes(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", _spy)

    cf.ingest_occurrences(project, source_csv, **kwargs)
    assert calls   # still read+hashed, even though nothing changed


def test_the_manifest_records_the_source_fingerprint(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    [manifest] = cf.load_imports(project).to_dict("records")
    assert manifest["import_source_path"] == str(source_csv)
    assert manifest["import_source_bytes"] == source_csv.stat().st_size
    assert manifest["import_source_mtime"] == source_csv.stat().st_mtime


def test_the_manifest_fingerprint_is_absent_for_an_in_memory_import(tmp_path):
    project = tmp_path / "project"
    missing_path = tmp_path / "nonexistent_synthetic.csv"
    df = pd.DataFrame({"detection_id": ["1"], "photo": ["http://x/1.jpg"]})
    cf.ingest_occurrences(project, missing_path, id_col="detection_id",
                          raw_bytes=b"synthetic", df=df)

    [manifest] = cf.load_imports(project).to_dict("records")
    assert pd.isna(manifest["import_source_bytes"])
    assert pd.isna(manifest["import_source_mtime"])


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
# import manifests: what happened to a raw import to make it an import
# ---------------------------------------------------------------------------


def test_the_manifest_records_structural_and_judgement_decisions(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                          image_url_col="photo", datetime_cols=["captured"],
                          drop={"determination_name": ["Debris"]})

    [manifest] = cf.load_imports(project).to_dict("records")
    assert manifest["id_col"] == "detection_id"          # structural
    assert manifest["image_url_col"] == "photo"           # structural
    assert manifest["drop"] == {"determination_name": ["Debris"]}   # judgement
    assert manifest["row_counts"] == {"read": 4, "dropped": 1, "capped": 0, "final": 3}


def test_the_manifest_names_a_transform_rather_than_its_repr(tmp_path, source_csv):
    def add_site(df):
        return df.assign(site=df[ID_COL].str[0])

    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id",
                          transform=add_site)

    [manifest] = cf.load_imports(project).to_dict("records")
    assert "add_site" in manifest["transform"]


def test_the_manifest_sits_beside_the_raw_import_it_describes(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")

    raw = imports_of(project)[0]
    manifests = list(paths.raw_imports_dir(project).glob("*.import.json"))
    assert len(manifests) == 1
    assert manifests[0].name.startswith(raw.stem)


def test_load_imports_is_empty_before_any_ingest(tmp_path):
    assert cf.load_imports(tmp_path / "project").empty


# ---------------------------------------------------------------------------
# group_col / max_per_group
# ---------------------------------------------------------------------------


@pytest.fixture
def lopsided_csv(tmp_path):
    """Seven of one species, three of another -- the shape a real pull takes."""
    path = tmp_path / "lopsided.csv"
    pd.DataFrame({
        "occurrence_id": [f"o{index}" for index in range(10)],
        "species": ["common"] * 7 + ["rare"] * 3,
    }).to_csv(path, index=False)
    return path


def test_max_per_group_caps_each_group_independently(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, lopsided_csv, group_col="species",
                                  max_per_group=2)

    counts = table["species"].value_counts()
    assert counts["common"] == 2
    assert counts["rare"] == 2


def test_group_col_without_max_per_group_raises(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    with pytest.raises(ValueError, match="group_col and max_per_group"):
        cf.ingest_occurrences(project, lopsided_csv, group_col="species")


def test_max_per_group_without_group_col_raises(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    with pytest.raises(ValueError, match="group_col and max_per_group"):
        cf.ingest_occurrences(project, lopsided_csv, max_per_group=2)


def test_the_cap_is_applied_after_drop(tmp_path):
    """
    A row drop= excludes as not-an-organism must never count toward its
    group's cap -- were the cap applied first, the excluded row would already
    have used up one of the group's slots.
    """
    source = tmp_path / "export.csv"
    pd.DataFrame({
        "occurrence_id": [f"o{index}" for index in range(5)],
        "species": ["moth"] * 5,
        "determination_name": ["Not Lepidoptera"] + ["moth"] * 4,
    }).to_csv(source, index=False)

    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source, drop={"determination_name": ["Not Lepidoptera"]},
        group_col="species", max_per_group=4, cap_rule="first")

    # drop removes o0 first, leaving exactly 4 -- right at the cap, untouched.
    # Capping before drop would instead keep o0..o3 by file order and then
    # drop o0, leaving only 3.
    assert table[ID_COL].tolist() == ["o1", "o2", "o3", "o4"]


def test_capped_rows_are_still_in_the_archive(tmp_path, lopsided_csv):
    """What makes capping safe: the import keeps everything the source sent."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, lopsided_csv, group_col="species",
                          max_per_group=2)

    assert len(pd.read_csv(imports_of(project)[0])) == 10


def test_reimporting_with_a_higher_cap_keeps_what_was_already_kept(
        tmp_path, lopsided_csv):
    """
    The point of keep_ids: a later pull growing max_per_group should be
    additive -- same specimens the project already has, plus more -- not a
    reshuffle that silently orphans work already done on specimens that are
    still perfectly good candidates.
    """
    project = tmp_path / "project"
    first = cf.ingest_occurrences(project, lopsided_csv, group_col="species",
                                  max_per_group=2)
    first_ids = set(first[ID_COL])

    second = cf.ingest_occurrences(project, lopsided_csv, group_col="species",
                                   max_per_group=4)
    assert first_ids <= set(second[ID_COL])
    assert second["species"].value_counts()["common"] == 4


def test_reimporting_with_a_lower_cap_retrims_by_rule(tmp_path):
    source = tmp_path / "export.csv"
    pd.DataFrame({
        "occurrence_id": [f"o{index}" for index in range(5)],
        "species": ["moth"] * 5,
    }).to_csv(source, index=False)

    project = tmp_path / "project"
    cf.ingest_occurrences(project, source, group_col="species",
                          max_per_group=4, cap_rule="first")
    second = cf.ingest_occurrences(project, source, group_col="species",
                                   max_per_group=2, cap_rule="first")

    assert second[ID_COL].tolist() == ["o0", "o1"]


def test_a_projects_first_ingest_has_nothing_to_prioritize(tmp_path, lopsided_csv):
    """No existing occurrences yet -- capping behaves exactly as it always did."""
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, lopsided_csv, group_col="species",
                                  max_per_group=2)
    assert table["species"].value_counts()["common"] == 2


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
    assert not list(paths.raw_imports_dir(project).glob("*.png"))
    assert not list(paths.raw_imports_dir(project).glob(".manifest.csv"))


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
