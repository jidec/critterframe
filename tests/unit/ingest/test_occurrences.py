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

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records.occurrences import ID_COL
from helpers.imports import imports_of


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
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", name_prefix="occurrences_antenna_199")

    name = imports_of(project)[0].stem
    assert name.startswith("occurrences_antenna_199_")
    datetime.date.fromisoformat(name.rsplit("_", 1)[-1])


def test_two_different_imports_on_one_day_do_not_clobber_each_other(tmp_path, source_csv, monkeypatch):
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
    pd.read_csv(source_csv).assign(detection_id=lambda df: df["detection_id"] + 10).to_csv(
        revised, index=False
    )
    cf.ingest_occurrences(project, revised, id_col="detection_id", name_prefix="occurrences")

    assert [path.name for path in imports_of(project)] == [
        "occurrences_2026-03-14.csv",
        "occurrences_2026-03-14_1.csv",
    ]


def test_reingesting_the_same_raw_import_the_same_way_is_a_no_op(tmp_path, source_csv):
    """
    The idempotency guarantee: identical raw content plus identical decisions
    is recognized as work already done, so a scheduled re-pull that finds
    nothing new doesn't re-archive, re-parse, or re-save anything.
    """
    project = tmp_path / "project"
    first = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Debris"]}
    )
    second = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Debris"]}
    )

    assert first[ID_COL].tolist() == second[ID_COL].tolist()
    assert len(imports_of(project)) == 1  # no duplicate archived


def test_reingesting_the_same_raw_import_a_different_way_is_not_a_no_op(tmp_path, source_csv):
    """
    Different decisions over identical content is a different IMPORT, even
    though it's the same raw import -- the whole point of the two-tier
    vocabulary. The raw content is still deduplicated: one raw file serves
    both.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", drop=None)
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Not Lepidoptera", "Debris"]}
    )

    assert table[ID_COL].tolist() == ["1", "3"]
    assert len(imports_of(project)) == 1  # same raw content, stored once
    assert len(cf.load_imports(project)) == 2  # two distinct imports


def test_a_repeat_is_recognized_by_content_and_decisions(tmp_path, source_csv):
    """
    The same raw bytes under the same decisions are one import however often
    they are ingested, and a different decision over the same bytes is another.
    """
    project = tmp_path / "project"
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})

    cf.ingest_occurrences(project, source_csv, **kwargs)
    cf.ingest_occurrences(project, source_csv, **kwargs)
    assert len(cf.load_imports(project)) == 1

    cf.ingest_occurrences(project, source_csv, id_col="detection_id", drop=None)
    assert len(cf.load_imports(project)) == 2


def test_ingest_normalizes_and_returns_the_table(tmp_path, source_csv):
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", image_url_col="photo", datetime_cols=["captured"]
    )

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
        project, source_csv, id_col="detection_id", transform=lambda df: df.assign(site=df[ID_COL].str[0])
    )
    assert "site" in table.columns


def test_drop_keeps_declared_non_organisms_out_of_the_table(tmp_path, source_csv):
    """
    The occurrence contract, not a filter: every row in this table asserts one
    focal organism, and a detection the source classified as debris asserts
    nothing.
    """
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Not Lepidoptera", "Debris"]}
    )

    assert table[ID_COL].tolist() == ["1", "3"]
    assert "Not Lepidoptera" not in set(table["determination_name"])


def test_the_dropped_rows_are_still_in_the_archive(tmp_path, source_csv):
    """What makes dropping safe: the import keeps everything the source sent."""
    project = tmp_path / "project"
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Not Lepidoptera"]}
    )

    assert len(pd.read_csv(imports_of(project)[0])) == 4


def test_drop_is_applied_after_the_transform(tmp_path, source_csv):
    """So a rule may name a column the transform joined in."""
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project,
        source_csv,
        id_col="detection_id",
        transform=lambda df: df.assign(verdict=df["determination_name"]),
        drop={"verdict": ["Debris"]},
    )
    assert "4" not in table[ID_COL].tolist()


def test_a_drop_rule_on_an_unknown_column_raises(tmp_path, source_csv):
    """
    A typo that silently matched nothing would read as "there was none of
    that here", which is the wrong answer to have believed.
    """
    project = tmp_path / "project"
    with pytest.raises(KeyError, match="rule column"):
        cf.ingest_occurrences(
            project, source_csv, id_col="detection_id", drop={"determinaton_name": ["Debris"]}
        )


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


def test_deduplication_runs_after_drop(tmp_path):
    """
    An ABSENT row must not be able to win a duplicate group and take the real
    sighting down with it -- which is what deduplicating before drop= did.
    """
    path = tmp_path / "duplicated.csv"
    pd.DataFrame(
        {
            "occurrence_id": ["absent_copy", "real"],
            "status": ["ABSENT", "PRESENT"],
            "lat": [1.0, 1.0],
            "lon": [2.0, 2.0],
        }
    ).to_csv(path, index=False)

    table = cf.ingest_occurrences(
        tmp_path / "project", path, drop={"status": ["ABSENT"]}, dedupe_key_cols=["lat", "lon"]
    )
    assert table["occurrence_id"].tolist() == ["real"]


# ---------------------------------------------------------------------------
# group_col / max_per_group
# ---------------------------------------------------------------------------


@pytest.fixture
def lopsided_csv(tmp_path):
    """Seven of one species, three of another -- the shape a real pull takes."""
    path = tmp_path / "lopsided.csv"
    pd.DataFrame(
        {
            "occurrence_id": [f"o{index}" for index in range(10)],
            "species": ["common"] * 7 + ["rare"] * 3,
        }
    ).to_csv(path, index=False)
    return path


def test_max_per_group_caps_each_group_independently(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=2)

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
    pd.DataFrame(
        {
            "occurrence_id": [f"o{index}" for index in range(5)],
            "species": ["moth"] * 5,
            "determination_name": ["Not Lepidoptera"] + ["moth"] * 4,
        }
    ).to_csv(source, index=False)

    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project,
        source,
        drop={"determination_name": ["Not Lepidoptera"]},
        group_col="species",
        max_per_group=4,
        cap_rule="first",
    )

    # drop removes o0 first, leaving exactly 4 -- right at the cap, untouched.
    # Capping before drop would instead keep o0..o3 by file order and then
    # drop o0, leaving only 3.
    assert table[ID_COL].tolist() == ["o1", "o2", "o3", "o4"]


def test_capped_rows_are_still_in_the_archive(tmp_path, lopsided_csv):
    """What makes capping safe: the import keeps everything the source sent."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=2)

    assert len(pd.read_csv(imports_of(project)[0])) == 10


def test_reimporting_with_a_higher_cap_keeps_what_was_already_kept(tmp_path, lopsided_csv):
    """
    The point of keep_ids: a later pull growing max_per_group should be
    additive -- same specimens the project already has, plus more -- not a
    reshuffle that silently orphans work already done on specimens that are
    still perfectly good candidates.
    """
    project = tmp_path / "project"
    first = cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=2)
    first_ids = set(first[ID_COL])

    second = cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=4)
    assert first_ids <= set(second[ID_COL])
    assert second["species"].value_counts()["common"] == 4


def test_reimporting_with_a_lower_cap_retrims_by_rule(tmp_path):
    source = tmp_path / "export.csv"
    pd.DataFrame(
        {
            "occurrence_id": [f"o{index}" for index in range(5)],
            "species": ["moth"] * 5,
        }
    ).to_csv(source, index=False)

    project = tmp_path / "project"
    cf.ingest_occurrences(project, source, group_col="species", max_per_group=4, cap_rule="first")
    second = cf.ingest_occurrences(project, source, group_col="species", max_per_group=2, cap_rule="first")

    assert second[ID_COL].tolist() == ["o0", "o1"]


def test_a_projects_first_ingest_has_nothing_to_prioritize(tmp_path, lopsided_csv):
    """No existing occurrences yet -- capping behaves exactly as it always did."""
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=2)
    assert table["species"].value_counts()["common"] == 2


# ---------------------------------------------------------------------------
# visualize=
# ---------------------------------------------------------------------------


def pipeline_files(project_path):
    directory = paths.pipeline_dir(project_path)
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def test_an_import_draws_its_stages_and_its_capped_groups(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, lopsided_csv, group_col="species", max_per_group=2)

    files = pipeline_files(project)
    assert any(name.endswith("__stages.png") for name in files)
    assert any(name.endswith("__groups.png") for name in files)


def test_a_skipped_reimport_draws_nothing_new(tmp_path, lopsided_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, lopsided_csv)
    before = pipeline_files(project)

    cf.ingest_occurrences(project, lopsided_csv)
    assert pipeline_files(project) == before


def test_stage_counts_are_not_part_of_the_import(tmp_path, lopsided_csv):
    """Descriptive only: a caller's own counts must not make an identical import look new."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, lopsided_csv, stage_counts={"fetched": 12})
    cf.ingest_occurrences(project, lopsided_csv, stage_counts={"fetched": 99})
    assert len(cf.load_imports(project)) == 1


def test_an_image_ingest_leaves_a_thumbnail_grid(tmp_path, image_dir):
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir)

    files = pipeline_files(project)
    assert any(name.startswith("ingest_images__images_") and name.endswith(".jpg") for name in files)


def test_ingest_visualize_false_writes_nothing(tmp_path, image_dir, lopsided_csv):
    project = tmp_path / "project"
    cf.ingest_images(project, image_dir, visualize=False)
    cf.ingest_occurrences(tmp_path / "other", lopsided_csv, visualize=False)
    assert pipeline_files(project) == []
    assert pipeline_files(tmp_path / "other") == []


# ---------------------------------------------------------------------------
# ingest_occurrences: prefer= (ranking inside dedupe and the cap)
# ---------------------------------------------------------------------------


def test_prefer_is_recorded_only_when_it_is_set(tmp_path, source_csv):
    """So every import archived before this option existed keeps its own hash."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    [manifest] = cf.load_imports(project).to_dict("records")
    assert "prefer" not in manifest
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", prefer=None)
    assert len(cf.load_imports(project)) == 1


def test_setting_prefer_is_a_different_import(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", prefer={"determination_name": ["Noctuidae"]}
    )
    imports = cf.load_imports(project)
    assert imports["import_hash"].nunique() == 2
    assert imports.iloc[-1]["prefer"] == {"determination_name": ["Noctuidae"]}


def test_prefer_reaches_both_dedupe_and_the_cap(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "occurrence_id": ["o1", "i1", "o2", "o3", "i2"],
            "source": ["other", "inat", "other", "other", "inat"],
            "species": ["common"] * 5,
            "lat": [1.0, 1.0, 3.0, 4.0, 5.0],  # o1 and i1 are one sighting
        }
    ).to_csv(path, index=False)

    table = cf.ingest_occurrences(
        tmp_path / "project",
        path,
        dedupe_key_cols=["lat"],
        group_col="species",
        max_per_group=2,
        prefer={"source": ["inat"]},
    )
    assert sorted(table[ID_COL]) == ["i1", "i2"]


# ---------------------------------------------------------------------------
# ingest_occurrences: no type inference on a CSV
# ---------------------------------------------------------------------------


def test_a_csv_is_read_without_guessing_types(tmp_path):
    path = tmp_path / "coded.csv"
    path.write_text("id,country,count,score\n007,NA,3,0.5\n010,,4,0.7\n")

    table = cf.ingest_occurrences(tmp_path / "project", path, id_col="id", numeric_cols=["score"])
    assert table[ID_COL].tolist() == ["007", "010"]
    assert table["country"].iloc[0] == "NA"  # a value, not a blank
    assert pd.isna(table["country"].iloc[1])  # only an empty field is missing
    assert table["count"].tolist() == ["3", "4"]  # not named, so not typed
    assert table["score"].dtype.kind == "f"


# ---------------------------------------------------------------------------
# ingest_occurrences: read= and raw=
# ---------------------------------------------------------------------------


def _recording_read(calls):
    def read_source(path):
        calls.append(path)
        return pd.read_csv(path, dtype=str), {"source rows": 99}

    return read_source


def test_read_runs_only_for_a_new_import(tmp_path, source_csv):
    calls = []
    read = _recording_read(calls)
    project = tmp_path / "project"

    cf.ingest_occurrences(project, source_csv, id_col="detection_id", read=read)
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", read=read)
    assert len(calls) == 1


def test_read_stage_counts_come_before_the_read(tmp_path, source_csv, monkeypatch):
    import critterframe.ingest.occurrences as core

    drawn = []
    original = core.figures.funnel

    def capture(stages, **kwargs):
        drawn.append(list(stages))
        return original(stages, **kwargs)

    monkeypatch.setattr(core.figures, "funnel", capture)
    cf.ingest_occurrences(tmp_path / "project", source_csv, id_col="detection_id", read=_recording_read([]))
    [stages] = drawn
    assert stages[:2] == ["source rows", "read"]


def test_raw_decides_what_is_archived(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", raw=lambda path: (b"the real source", ".bin")
    )
    [archived] = paths.raw_imports_dir(project).glob("*.bin")
    assert archived.read_bytes() == b"the real source"


def test_a_directory_source_is_never_trusted_on_its_fingerprint(tmp_path, source_csv):
    """A directory's size says nothing about its contents."""
    directory = tmp_path / "extracted"
    directory.mkdir()
    (directory / "table.csv").write_bytes(source_csv.read_bytes())
    calls = []
    kwargs = dict(
        id_col="detection_id",
        read=lambda d: _recording_read(calls)(d / "table.csv"),
        raw=lambda d: ((d / "table.csv").read_bytes(), ".csv"),
        trust_source_file_unchanged=True,
    )
    project = tmp_path / "project"

    cf.ingest_occurrences(project, directory, **kwargs)
    (directory / "table.csv").write_text("detection_id,photo\n9,http://x/9.jpg\n")
    table = cf.ingest_occurrences(project, directory, **kwargs)
    assert table[ID_COL].tolist() == ["9"]
