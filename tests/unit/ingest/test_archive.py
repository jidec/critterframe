"""
The raw-import archive: bytes stored once, and found again after a project moves.
"""

import critterframe as cf
from critterframe.project import paths
from helpers.imports import imports_of


# ---------------------------------------------------------------------------
# A copied project still finds its own raw imports
# ---------------------------------------------------------------------------


def test_the_raw_import_is_logged_relative_to_the_project(tmp_path, source_csv):
    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    [manifest] = cf.load_imports(project).to_dict("records")
    assert manifest["raw_path"].startswith("raw_imports/")


def test_a_copied_project_reuses_its_archived_raw_import(tmp_path, source_csv):
    import shutil

    original = tmp_path / "original"
    cf.ingest_occurrences(original, source_csv, id_col="detection_id")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(original)

    cf.ingest_occurrences(moved, source_csv, id_col="detection_id", drop={"determination_name": ["Debris"]})
    assert len(imports_of(moved)) == 1
    assert cf.load_imports(moved)["raw_import_reused"].tolist() == [False, True]


def test_a_legacy_absolute_raw_path_resolves_by_name(tmp_path, source_csv):
    """An older log, possibly written on another OS, holding an absolute path."""
    import json

    project = tmp_path / "project"
    cf.ingest_occurrences(project, source_csv, id_col="detection_id")
    log = paths.imports_log_path(project)
    [record] = [json.loads(line) for line in log.read_text().splitlines()]
    name = record["raw_path"].rsplit("/", 1)[-1]
    record["raw_path"] = r"C:\Users\someone\old_place\raw_imports" + "\\" + name
    log.write_text(json.dumps(record) + "\n")

    cf.ingest_occurrences(project, source_csv, id_col="detection_id", drop={"determination_name": ["Debris"]})
    assert len(imports_of(project)) == 1
    assert cf.load_imports(project)["raw_import_reused"].iloc[-1]
