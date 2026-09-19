"""
Run records, and the migration that rewrites an old project on first open.

`_drop_legacy_metric_columns` is the one destructive, unrecoverable code path in
the package. It runs automatically, on every open, against any database written
before those columns were removed -- and until now nothing exercised it at all.
It has to run: the old `created_at` was NOT NULL, so a database still carrying
it rejects every new insert. The test below builds that old schema by hand and
checks both halves: the columns go, and writing works afterward.

The additive migration beside it is gentler -- a nullable column an old database
merely lacks -- but it has to run for the same reason: start_run names the
column whether or not the database has it yet.
"""

import json
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


def test_runs_can_be_filtered_by_run_id(tmp_path):
    first = run_records.start_run(tmp_path, a_recipe(name="run0"))
    run_records.start_run(tmp_path, a_recipe(name="run1"))

    found = run_records.load_runs(tmp_path, run_id=first)
    assert found["name"].tolist() == ["run0"]
    assert run_records.load_runs(tmp_path, run_id=999).empty


def test_a_project_with_no_runs_reads_as_empty(tmp_path):
    assert run_records.load_runs(tmp_path).empty


# ---------------------------------------------------------------------------
# The run context: what a run covered, beside what it was
# ---------------------------------------------------------------------------


def test_a_run_records_what_it_covered(tmp_path):
    """
    The recipe says what the work was; the context says what it was done to.
    Neither is derivable from the other, and only the recipe is hashed.
    """
    context = {"occurrences": {"count": 40, "ids_hash": "abcd"}, "limit": None}
    run_records.start_run(tmp_path, a_recipe(), context=context)

    assert run_records.load_runs(tmp_path)["context"].iloc[0] == context


def test_a_run_with_nothing_to_say_stores_nothing(tmp_path):
    """No context is None, not an empty dict -- the column is genuinely unset."""
    run_records.start_run(tmp_path, a_recipe())
    assert run_records.load_runs(tmp_path)["context"].iloc[0] is None


def test_a_database_written_before_the_context_column_gains_it(tmp_path):
    """
    Adding a nullable column is not like dropping the legacy NOT NULL one: an
    old database is merely missing it rather than broken by it. But start_run
    names the column, so an unmigrated database would reject every new run.
    """
    from critterframe.storage.sqlite import connect

    connection = connect(paths.runs_and_metrics_path(tmp_path))
    connection.execute(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
            part TEXT NOT NULL, subset TEXT, recipe_hash TEXT NOT NULL,
            recipe_json TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, finished_at TEXT,
            n_processed INTEGER NOT NULL DEFAULT 0,
            n_skipped INTEGER NOT NULL DEFAULT 0,
            n_failed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "INSERT INTO runs (kind, name, part, recipe_hash, recipe_json, status, "
        "created_at) VALUES ('metric','old','organism','abc','{}','complete','2020-01-01')"
    )
    connection.commit()
    connection.close()

    run_records.start_run(tmp_path, a_recipe(name="new"),
                          context={"occurrences": {"count": 1}})

    runs = run_records.load_runs(tmp_path).set_index("name")
    assert runs.loc["old", "context"] is None       # nothing to recover
    assert runs.loc["new", "context"] == {"occurrences": {"count": 1}}


# ---------------------------------------------------------------------------
# Recipe currency: the metric-side pointer masks get for free from upserting
# ---------------------------------------------------------------------------


def test_segment_kind_is_always_a_no_op(tmp_path):
    """
    Masks.parquet upserts to a single current row per occurrence-part, so a
    segment run_name pointing at several hashes over a project's life is
    exactly what resegmenting is, not ambiguity to guard against.
    """
    first = run_records.resolve_recipe_currency(
        tmp_path, "segment", "segments", "organism", "hash_a", force=False)
    second = run_records.resolve_recipe_currency(
        tmp_path, "segment", "segments", "organism", "hash_b", force=False)
    assert (first, second) == (False, False)
    assert run_records.current_recipe_pointers(tmp_path, kind="segment") == {}


def test_first_use_adopts_silently(tmp_path):
    needs_commit = run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_a", force=False)
    assert needs_commit is False
    assert run_records.current_recipe_pointers(tmp_path) == {
        ("traits", "organism"): "hash_a"}


def test_the_same_hash_again_is_never_a_conflict(tmp_path):
    """An identical rerun -- retrying after an interruption, say -- has
    nothing to acknowledge, whether or not force is passed."""
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_a", force=False)
    for force in (False, True):
        needs_commit = run_records.resolve_recipe_currency(
            tmp_path, "metric", "traits", "organism", "hash_a", force=force)
        assert needs_commit is False


def test_a_different_hash_without_force_raises(tmp_path):
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_a", force=False)
    with pytest.raises(ValueError, match="currently points at a different"):
        run_records.resolve_recipe_currency(
            tmp_path, "metric", "traits", "organism", "hash_b", force=False)
    # Refused, so nothing moved.
    assert run_records.current_recipe_pointers(tmp_path) == {
        ("traits", "organism"): "hash_a"}


def test_a_different_hash_with_force_does_not_move_the_pointer_by_itself(tmp_path):
    """
    resolve_recipe_currency only checks and reports what's owed -- it doesn't
    write a forced change itself, because the caller hasn't yet confirmed the
    new recipe produced anything.
    """
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_a", force=False)
    needs_commit = run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_b", force=True)
    assert needs_commit is True
    assert run_records.current_recipe_pointers(tmp_path) == {
        ("traits", "organism"): "hash_a"}


def test_commit_recipe_currency_moves_the_pointer(tmp_path):
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_a", force=False)
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", "hash_b", force=True)
    run_records.commit_recipe_currency(tmp_path, "metric", "traits",
                                       "organism", "hash_b")
    assert run_records.current_recipe_pointers(tmp_path) == {
        ("traits", "organism"): "hash_b"}


def test_commit_recipe_currency_is_a_no_op_for_segment(tmp_path):
    run_records.commit_recipe_currency(tmp_path, "segment", "segments",
                                       "organism", "hash_a")
    assert run_records.current_recipe_pointers(tmp_path, kind="segment") == {}


def test_parts_are_independent(tmp_path):
    """A shared run_name across parts -- run_segments' outputs= pattern, ported
    to metrics -- never conflicts, since each part is its own key."""
    run_records.resolve_recipe_currency(
        tmp_path, "metric", "body_parts", "head", "hash_head", force=False)
    needs_commit = run_records.resolve_recipe_currency(
        tmp_path, "metric", "body_parts", "abdomen", "hash_abdomen", force=False)
    assert needs_commit is False
    assert run_records.current_recipe_pointers(tmp_path) == {
        ("body_parts", "head"): "hash_head",
        ("body_parts", "abdomen"): "hash_abdomen",
    }


def test_an_existing_project_s_history_seeds_the_pointer(tmp_path):
    """
    A project with runs recorded from before this pointer existed must not
    have its next call silently adopt whatever hash happens to run next --
    it's seeded from the most recent run in history, so a rerun of THAT
    recipe stays silent and only a genuinely different one is caught.
    """
    run_records.start_run(tmp_path, a_recipe(name="traits"))   # hash from a_recipe()
    old_hash = a_recipe(name="traits").hash

    # No pointer row exists yet -- only run history, as an old database would
    # have. The same hash running again must not raise.
    needs_commit = run_records.resolve_recipe_currency(
        tmp_path, "metric", "traits", "organism", old_hash, force=False)
    assert needs_commit is False

    # A genuinely different hash is still caught, seeded from that history.
    run_records.start_run(tmp_path, a_recipe(name="fresh"))
    with pytest.raises(ValueError, match="currently points at a different"):
        run_records.resolve_recipe_currency(
            tmp_path, "metric", "fresh", "organism", "hash_b", force=False)


# ---------------------------------------------------------------------------
# runs.jsonl: a human-readable mirror of finish_run, for grep/jq without
# opening the database -- the same reasoning exports.jsonl/imports.jsonl exist.
# ---------------------------------------------------------------------------


def _read_log(project_path):
    log = paths.runs_log_path(project_path)
    if not log.exists():
        return []
    with open(log, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_finishing_a_run_appends_to_the_log(tmp_path):
    run_id = run_records.start_run(tmp_path, a_recipe(name="traits"))
    run_records.finish_run(tmp_path, run_id, processed=3)

    entries = _read_log(tmp_path)
    assert len(entries) == 1
    assert entries[0]["run_id"] == run_id
    assert entries[0]["name"] == "traits"
    assert entries[0]["n_processed"] == 3


def test_the_log_entry_s_recipe_and_context_are_parsed_back(tmp_path):
    """
    Same shape load_runs() hands back -- dicts, not the raw JSON strings the
    sqlite row stores them as.
    """
    recipe = a_recipe(name="traits")
    context = {"occurrences": {"count": 4}}
    run_id = run_records.start_run(tmp_path, recipe, context=context)
    run_records.finish_run(tmp_path, run_id)

    entry = _read_log(tmp_path)[0]
    assert entry["recipe"] == recipe.spec()
    assert entry["context"] == context
    assert "recipe_json" not in entry
    assert "context_json" not in entry


def test_a_run_left_running_is_not_logged(tmp_path):
    """An interrupted process that never reaches finish_run keeps its sqlite
    row but leaves no log line -- the log is "calls that completed", not
    "runs that started"."""
    run_records.start_run(tmp_path, a_recipe())
    assert _read_log(tmp_path) == []


def test_two_finished_runs_append_two_lines_in_order(tmp_path):
    first = run_records.start_run(tmp_path, a_recipe(name="run0"))
    second = run_records.start_run(tmp_path, a_recipe(name="run1"))
    run_records.finish_run(tmp_path, first)
    run_records.finish_run(tmp_path, second)

    assert [entry["run_id"] for entry in _read_log(tmp_path)] == [first, second]


def test_a_project_with_no_finished_runs_has_no_log_file(tmp_path):
    assert not paths.runs_log_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# the database file itself
# ---------------------------------------------------------------------------


def test_reading_a_project_that_never_ran_creates_no_database(tmp_path):
    """
    paths promises it creates nothing, and a summary or an export asking what a
    project holds is a read -- it shouldn't leave a database behind in one that
    has never run anything.
    """
    from critterframe.project import paths

    assert run_records.load_runs(tmp_path).empty
    assert run_records.current_recipe_pointers(tmp_path) == {}
    assert not paths.runs_and_metrics_path(tmp_path).exists()


def test_an_empty_run_table_still_has_its_columns(tmp_path):
    """So a caller can filter or read a column without special-casing empty."""
    runs = run_records.load_runs(tmp_path)
    assert "recipe_hash" in runs.columns and "context" in runs.columns


def test_a_read_does_not_hold_the_database_open(tmp_path):
    """
    An unclosed connection keeps a file handle for the life of the process,
    which on Windows is enough to stop the project directory being moved.
    """
    import os

    recipe = Recipe("metric", "traits", [body_length()], part="organism")
    run_records.start_run(tmp_path, recipe)
    run_records.load_runs(tmp_path)

    moved = tmp_path.parent / (tmp_path.name + "_moved")
    os.rename(tmp_path, moved)          # raises if anything still holds it open
    assert (moved / "runs_and_metrics.sqlite").exists()
