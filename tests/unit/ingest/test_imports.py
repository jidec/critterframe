"""
The import record: what an import is hashed from, what its manifest says, and
when a repeat is recognized.
"""

import pandas as pd

import critterframe as cf
from critterframe.project import paths
from critterframe.records.occurrences import ID_COL
from helpers.imports import imports_of


# ---------------------------------------------------------------------------
# trust_source_file_unchanged
# ---------------------------------------------------------------------------


def test_trust_source_file_unchanged_skips_the_read_on_a_repeat_call(tmp_path, source_csv, monkeypatch):
    """
    Same path, size, and mtime as a previous import: the content is never
    read again, only the cheap fingerprint is checked.
    """
    from pathlib import Path

    project = tmp_path / "project"
    kwargs = dict(id_col="detection_id", drop={"determination_name": ["Debris"]})
    first = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True, **kwargs)

    def _raise(self, *args, **kwargs):
        raise AssertionError("read_bytes should not be called on a fingerprint hit")

    monkeypatch.setattr(Path, "read_bytes", _raise)

    second = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True, **kwargs)
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

    table = cf.ingest_occurrences(project, source_csv, trust_source_file_unchanged=True, **kwargs)
    assert "5" in table[ID_COL].tolist()
    assert len(cf.load_imports(project)) == 2


def test_trust_source_file_unchanged_still_reingests_on_a_changed_decision(tmp_path, source_csv):
    """
    The file's fingerprint matches, but drop= changed -- the borrowed
    raw_hash produces a different import_hash, so this correctly does NOT
    skip, unlike a naive "same file, always skip" shortcut would.
    """
    project = tmp_path / "project"
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", trust_source_file_unchanged=True, drop=None
    )
    table = cf.ingest_occurrences(
        project,
        source_csv,
        id_col="detection_id",
        trust_source_file_unchanged=True,
        drop={"determination_name": ["Not Lepidoptera", "Debris"]},
    )

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
    assert calls  # still read+hashed, even though nothing changed


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
    cf.ingest_occurrences(project, missing_path, id_col="detection_id", raw_bytes=b"synthetic", df=df)

    [manifest] = cf.load_imports(project).to_dict("records")
    assert pd.isna(manifest["import_source_bytes"])
    assert pd.isna(manifest["import_source_mtime"])


# ---------------------------------------------------------------------------
# import manifests: what happened to a raw import to make it an import
# ---------------------------------------------------------------------------


def test_the_manifest_records_structural_and_judgement_decisions(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(
        project,
        source_csv,
        id_col="detection_id",
        image_url_col="photo",
        datetime_cols=["captured"],
        drop={"determination_name": ["Debris"]},
    )

    [manifest] = cf.load_imports(project).to_dict("records")
    assert manifest["id_col"] == "detection_id"  # structural
    assert manifest["image_url_col"] == "photo"  # structural
    assert manifest["drop"] == {"determination_name": ["Debris"]}  # judgement
    assert manifest["row_counts"] == {"read": 4, "dropped": 1, "deduped": 0, "capped": 0, "final": 3}


def test_the_manifest_names_a_transform_rather_than_its_repr(tmp_path, source_csv):
    def add_site(df):
        return df.assign(site=df[ID_COL].str[0])

    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", transform=add_site)

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
# the decisions an import is hashed from
# ---------------------------------------------------------------------------


def add_one(df):
    return df.assign(derived=1)


def add_two(df):
    return df.assign(derived=2)


def test_a_sequence_of_transforms_is_applied_and_recorded_by_name(tmp_path, source_csv):
    """
    What an extension stacking its own derivations on a caller's needs. A
    closure around both would record only the wrapper's name, so two
    different callers' transforms would share an import hash.
    """
    project = tmp_path / "project"
    table = cf.ingest_occurrences(project, source_csv, id_col="detection_id", transform=[add_one, add_two])

    assert set(table["derived"]) == {2}  # applied in order
    [manifest] = cf.load_imports(project).to_dict("records")
    assert manifest["transform"] == ["add_one", "add_two"]


def test_two_different_transforms_are_two_different_imports(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", transform=add_one)
    cf.ingest_occurrences(project, source_csv, id_col="detection_id", transform=add_two)

    imports = cf.load_imports(project)
    assert len(imports) == 2
    assert imports["import_hash"].nunique() == 2


def test_a_set_of_drop_values_is_accepted(tmp_path, source_csv):
    """rows_matching takes a set, so hashing the decision has to as well."""
    project = tmp_path / "project"
    table = cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": {"Debris", "Not Lepidoptera"}}
    )
    assert len(table) == 2


def test_the_order_drop_values_are_written_in_does_not_matter(tmp_path, source_csv):
    """Otherwise the same decision, typed two ways, re-ingests the same file."""
    project = tmp_path / "project"
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Debris", "Not Lepidoptera"]}
    )
    cf.ingest_occurrences(
        project, source_csv, id_col="detection_id", drop={"determination_name": ["Not Lepidoptera", "Debris"]}
    )

    assert len(cf.load_imports(project)) == 1


def test_deduplication_is_recorded_only_when_it_is_on(tmp_path, source_csv):
    """So every import archived before this stage existed keeps its own hash."""
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    [manifest] = cf.load_imports(project).to_dict("records")
    assert "dedupe" not in manifest
