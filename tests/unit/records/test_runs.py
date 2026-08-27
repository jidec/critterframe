"""
Run records, and the migration that rewrites an old project on first open.

`_drop_legacy_metric_columns` is the one destructive, unrecoverable code path in
the package. It runs automatically, on every open, against any database written
before those columns were removed -- and until now nothing exercised it at all.
It has to run: the old `created_at` was NOT NULL, so a database still carrying
it rejects every new insert. The test below builds that old schema by hand and
checks both halves: the columns go, and writing works afterward.
"""

import sqlite3

import pytest

from critterframe.project import paths
from critterframe.recipes import Recipe
from critterframe.records import metrics as metric_records
from critterframe.records import runs as run_records
from critterframe.metrics.dimensions import body_length
from helpers.compare import is_iso_utc


def a_recipe(kind="metric", name="traits", part="organism"):
    return Recipe(kind, name, [body_length()], part=part)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_opening_creates_both_tables(tmp_path):
    """
    Whichever module opens the database first creates both tables, so neither
    records module has to care about ordering.
    """
    with run_records.open_database(tmp_path) as connection:
        tables = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "metrics"} <= tables


def test_opening_twice_is_harmless(tmp_path):
    """
    ensure_schema runs on every open, so it has to be idempotent -- and a
    second open must not discard the first's rows.
    """
    run_id = run_records.start_run(tmp_path, a_recipe())
    with run_records.open_database(tmp_path):
        pass
    assert len(run_records.load_runs(tmp_path)) == 1
    assert run_id == 1


def test_a_new_database_has_no_legacy_columns(tmp_path):
    with run_records.open_database(tmp_path) as connection:
        columns = {row["name"] for row in
                   connection.execute("PRAGMA table_info(metrics)")}
    assert columns.isdisjoint(run_records.LEGACY_METRIC_COLUMNS)


def test_an_old_database_is_migrated_and_writable(tmp_path):
    """
    The whole point of the migration. Build the old schema -- `version` plus a
    NOT NULL `created_at` -- put a row in it, then open it through the package
    and assert that the columns are gone, the row survived, and a fresh insert
    (which could not have filled created_at) succeeds.
    """
    database = paths.runs_and_metrics_path(tmp_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
            part TEXT NOT NULL, subset TEXT, recipe_hash TEXT NOT NULL,
            recipe_json TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, finished_at TEXT,
            n_processed INTEGER NOT NULL DEFAULT 0,
            n_skipped INTEGER NOT NULL DEFAULT 0,
            n_failed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE metrics (
            metric_id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL,
            occurrence_id TEXT NOT NULL, part TEXT NOT NULL,
            metric_name TEXT NOT NULL, value_json TEXT NOT NULL, unit TEXT,
            recipe_hash TEXT NOT NULL, source_mask_hash TEXT,
            version TEXT, created_at TEXT NOT NULL
        );
        INSERT INTO runs (kind, name, part, recipe_hash, recipe_json, status,
                          created_at)
        VALUES ('metric', 'old', 'organism', 'oldhash', '{}', 'complete',
                '2024-01-01T00:00:00+00:00');
        INSERT INTO metrics (run_id, occurrence_id, part, metric_name,
                             value_json, unit, recipe_hash, version, created_at)
        VALUES (1, 'a', 'organism', 'body_length', '12.0', 'px', 'oldhash',
                '1', '2024-01-01T00:00:00+00:00');
        """
    )
    legacy.commit()
    legacy.close()

    with run_records.open_database(tmp_path) as connection:
        columns = {row["name"] for row in
                   connection.execute("PRAGMA table_info(metrics)")}
    assert columns.isdisjoint(run_records.LEGACY_METRIC_COLUMNS)

    # The value itself is not what the migration drops.
    stored = metric_records.load_metrics(tmp_path)
    assert stored["value"].tolist() == [12.0]

    # And the insert the old NOT NULL column would have rejected now works.
    run_id = run_records.start_run(tmp_path, a_recipe())
    metric_records.append_metrics(
        tmp_path, run_id, "newhash",
        [metric_records.make_metric_row("b", "organism", "body_length", 3.5)])
    assert len(metric_records.load_metrics(tmp_path)) == 2


# ---------------------------------------------------------------------------
# start_run / finish_run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["render", "export", "", "Segment"])
def test_only_run_kinds_get_a_run_record(tmp_path, kind):
    """
    A render derives no data, so the hash naming its folder is the whole of its
    provenance and there is nothing to keep a run record for. The database says
    so rather than trusting callers.
    """
    with pytest.raises(ValueError, match="run kind must be one of"):
        run_records.start_run(tmp_path, a_recipe(kind=kind))


def test_a_started_run_records_what_it_meant_to_do(tmp_path):
    """
    The recipe spec is written at the START, so an interrupted run still says
    what it was attempting.
    """
    recipe = a_recipe()
    run_id = run_records.start_run(tmp_path, recipe, subset="amnh")

    row = run_records.load_runs(tmp_path).iloc[0]
    assert row["run_id"] == run_id
    assert row["status"] == run_records.STATUS_RUNNING
    assert row["finished_at"] is None
    assert row["recipe_hash"] == recipe.hash
    assert row["recipe"] == recipe.spec()
    assert row["subset"] == "amnh"
    assert is_iso_utc(row["created_at"])


def test_the_subset_is_recorded_but_not_hashed(tmp_path):
    """
    Which occurrences a recipe ran over is a property of the run. Processing
    the rest of the project later must continue the same work, not count as
    different work.
    """
    recipe = a_recipe()
    run_records.start_run(tmp_path, recipe, subset="amnh")
    run_records.start_run(tmp_path, recipe, subset="mcz")

    runs = run_records.load_runs(tmp_path)
    assert set(runs["subset"]) == {"amnh", "mcz"}
    assert runs["recipe_hash"].nunique() == 1


def test_finishing_records_the_counts(tmp_path):
    run_id = run_records.start_run(tmp_path, a_recipe())
    run_records.finish_run(tmp_path, run_id, processed=5, skipped=2, failed=1)

    row = run_records.load_runs(tmp_path).iloc[0]
    assert (row["n_processed"], row["n_skipped"], row["n_failed"]) == (5, 2, 1)
    assert row["status"] == run_records.STATUS_COMPLETE
    assert is_iso_utc(row["finished_at"])


def test_a_run_that_blew_up_is_recorded_as_failed(tmp_path):
    run_id = run_records.start_run(tmp_path, a_recipe())
    run_records.finish_run(tmp_path, run_id, status=run_records.STATUS_FAILED)
    assert run_records.load_runs(tmp_path).iloc[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------


def test_runs_come_back_newest_first(tmp_path):
    """
    Ordered by time then by run_id. The tiebreak is what matters here: three
    runs started in the same microsecond order by insertion, which is the only
    thing that can separate them.
    """
    for index in range(3):
        run_records.start_run(tmp_path, a_recipe(name=f"run{index}"))
    assert run_records.load_runs(tmp_path)["run_id"].tolist() == [3, 2, 1]


def test_runs_can_be_filtered(tmp_path):
    metric = a_recipe(kind="metric", name="traits")
    segment = Recipe("segment", "organisms", [], part="organism")
    run_records.start_run(tmp_path, metric)
    run_records.start_run(tmp_path, segment)

    assert run_records.load_runs(tmp_path, kind="segment")["name"].tolist() == ["organisms"]
    assert run_records.load_runs(tmp_path, name="traits")["kind"].tolist() == ["metric"]
    assert len(run_records.load_runs(tmp_path, recipe_hash=metric.hash)) == 1
    assert run_records.load_runs(tmp_path, recipe_hash="nothing").empty


def test_a_project_with_no_runs_reads_as_empty(tmp_path):
    assert run_records.load_runs(tmp_path).empty
