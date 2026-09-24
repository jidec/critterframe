"""
archive_project: a copy of a project fit to deposit.

Everything that says what was done stays; the images, raw source data and working
files go; and no record names a place on the machine that made it.
"""

import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths


@pytest.fixture
def archived(measured_project, tmp_path):
    cf.export_metrics(measured_project)
    scripts = tmp_path / "pipeline"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('segment and measure')\n")
    dest = cf.archive_project(measured_project, tmp_path / "deposit",
                              source_doi="10.15468/dl.example", scripts=scripts)
    return measured_project, dest


def test_images_raw_data_and_working_files_are_left_out(archived):
    _project, dest = archived
    for name in (paths.IMAGES_DIR, paths.MASK_SHARDS_DIR, paths.FAILURES_FILE,
                 paths.VISUALIZATIONS_DIR):
        assert not (dest / name).exists()
    raw = dest / paths.RAW_IMPORTS_DIR
    if raw.exists():
        assert all(path.suffix in (".json", ".jsonl") for path in raw.iterdir())


def test_the_record_of_what_was_done_is_kept(archived):
    project, dest = archived
    for name in (paths.OCCURRENCES_FILE, paths.MASKS_FILE, paths.RUNS_LOG_FILE,
                 "environment.json", "README.md", "code/run.py"):
        assert (dest / name).exists(), name
    assert list((dest / paths.EXPORTS_DIR).glob("*.export.json"))


def test_the_database_is_one_complete_file(archived):
    project, dest = archived
    query = "SELECT COUNT(*) FROM metrics"
    with closing(sqlite3.connect(paths.runs_and_metrics_path(project))) as original, \
            closing(sqlite3.connect(dest / paths.RUNS_AND_METRICS_FILE)) as copy:
        assert copy.execute(query).fetchone() == original.execute(query).fetchone()
        assert copy.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_source_paths_are_reduced_to_file_names(archived):
    project, dest = archived
    original = pd.read_parquet(paths.occurrences_path(project))["source_path"]
    archived_paths = pd.read_parquet(dest / paths.OCCURRENCES_FILE)["source_path"]
    assert all("/" in value or "\\" in value for value in original)
    assert list(archived_paths) == [value.replace("\\", "/").rsplit("/", 1)[-1]
                                    for value in original]


def test_no_record_names_the_machine_that_made_it(archived):
    project, dest = archived
    local = str(paths.project_dir(project).resolve().parent)
    for path in dest.rglob("*"):
        if path.suffix in (".json", ".jsonl", ".csv", ".md"):
            text = path.read_text(encoding="utf-8")
            assert local not in text and local.replace("\\", "/") not in text, path


def test_the_readme_cites_the_source_and_the_environment_is_recorded(archived):
    _project, dest = archived
    assert "10.15468/dl.example" in (dest / "README.md").read_text(encoding="utf-8")
    environment = json.loads((dest / "environment.json").read_text(encoding="utf-8"))
    assert environment["critterframe"] == cf.__version__
    assert environment["python"] and "numpy" in environment["packages"]


def test_a_windows_path_is_reduced_on_any_platform(measured_project, tmp_path):
    """A record written on Windows and archived on Linux or macOS, and the reverse."""
    log = paths.imports_log_path(measured_project)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"import_source_path": r"C:\Users\someone\data\pull.zip",
                                 "extra": {"source": "/home/someone/data/pull.zip"}}) + "\n")

    dest = cf.archive_project(measured_project, tmp_path / "deposit")
    last = (dest / paths.RAW_IMPORTS_DIR / paths.IMPORTS_LOG_FILE).read_text(
        encoding="utf-8").splitlines()[-1]
    assert json.loads(last) == {"import_source_path": "pull.zip",
                                "extra": {"source": "pull.zip"}}


def test_a_non_empty_destination_is_refused(measured_project, tmp_path):
    dest = tmp_path / "deposit"
    dest.mkdir()
    (dest / "something.txt").write_text("already here")
    with pytest.raises(FileExistsError):
        cf.archive_project(measured_project, dest)


def test_a_destination_inside_the_project_is_refused(measured_project):
    with pytest.raises(ValueError, match="inside the project"):
        cf.archive_project(measured_project, paths.project_dir(measured_project) / "deposit")
