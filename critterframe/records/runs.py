"""
The sqlite schema for run + metric records, and the migrations that keep older
databases readable.

A run is one execution of a recipe over a set of occurrences. Owns all three
tables, since a metric row references its run: `runs` and `metrics` are the
immutable history; `current_recipes` is the metric-side pointer that says
which recipe a run_name presently means (see resolve_recipe_currency).
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..recipes import canonical_json, load_json
from ..storage.jsonfiles import append_jsonl
from ..storage.sqlite import connect

logger = logging.getLogger(__name__)

# The recipe kinds that execute as runs. Recipe itself is open-ended (a render
# is a hashable operation chain too -- see visualization.products), but only
# work that DERIVES something gets a run record: a render produces pictures, and
# the hash naming its folder is the whole of its provenance.
RUN_KINDS = ("segment", "metric")

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


LEGACY_METRIC_COLUMNS = ("version", "created_at")

# Columns added to `runs` after projects existed. Nullable, so unlike the legacy
# metric columns an old database is merely missing them rather than broken by
# them -- but start_run and load_runs both name them, so they have to be there.
ADDED_RUN_COLUMNS = {"context_json": "TEXT"}


def _add_missing_run_columns(connection):
    """Add any `runs` column introduced after this project's database was created."""
    stored = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
    for column, declaration in ADDED_RUN_COLUMNS.items():
        if column not in stored:
            logger.info("migrating runs table: adding '%s' column", column)
            connection.execute(
                f"ALTER TABLE runs ADD COLUMN {column} {declaration}")


def _drop_legacy_metric_columns(connection):
    """
    Bring a metrics table written before those columns were removed into line
    with the schema above.

    This migrates rather than tolerating both shapes because the old created_at
    was NOT NULL: a database still carrying it doesn't just hold dead weight, it
    rejects every insert that doesn't fill it. Dropping the column is what lets
    an existing project keep running.

    Neither column's contents are recoverable afterward. Both were duplicates --
    the recipe spec on each row's run carries the operation version, and the run
    carries the time -- so what's lost is per-row timing WITHIN a long run,
    which nothing here ever read.
    """
    stored = {row["name"] for row in connection.execute("PRAGMA table_info(metrics)")}
    for column in LEGACY_METRIC_COLUMNS:
        if column in stored:
            logger.info("migrating metrics table: dropping legacy '%s' column",
                        column)
            connection.execute(f"ALTER TABLE metrics DROP COLUMN {column}")


def ensure_schema(connection):
    """
    Create the runs and metrics tables if they don't exist yet.

    Both tables are created together, by whichever of the two modules opens the
    database first, so neither has to care about ordering. Flexible sections
    (the recipe, a metric's value) are stored as JSON so new operations and new
    metric shapes don't require a schema change -- which is exactly the freedom
    a package whose whole point is composable, user-defined recipes needs.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            part TEXT NOT NULL,
            subset TEXT,
            recipe_hash TEXT NOT NULL,
            recipe_json TEXT NOT NULL,
            status TEXT NOT NULL,
            context_json TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            n_processed INTEGER NOT NULL DEFAULT 0,
            n_skipped INTEGER NOT NULL DEFAULT 0,
            n_failed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _add_missing_run_columns(connection)
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS runs_recipe_idx
        ON runs (recipe_hash, created_at)
        """
    )
    # No per-row version or timestamp: both were duplicates of what the run
    # already carries. A metric operation's version is in the recipe spec, and
    # therefore in the recipe hash on every row; a value's time is its run's
    # created_at. metric_id is the table's own insertion order, which orders
    # values more truthfully than a timestamp shared by every row of one write.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            occurrence_id TEXT NOT NULL,
            part TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value_json TEXT NOT NULL,
            unit TEXT,
            recipe_hash TEXT NOT NULL,
            source_mask_hash TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
        """
    )
    _drop_legacy_metric_columns(connection)
    # The repeat-awareness index: "has this recipe already covered this
    # occurrence-part" is the query every metric run issues before doing any
    # work, so it has to stay cheap as the table grows.
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS metrics_recipe_idx
        ON metrics (recipe_hash, occurrence_id, part)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS metrics_occurrence_idx
        ON metrics (occurrence_id, part, metric_name)
        """
    )
    # The metric-side counterpart of what masks.parquet already gives
    # segmentation for free: one designated "current" recipe per (kind, name,
    # part), separate from the immutable run history above. Masks don't need
    # this table -- upserting to a single row per occurrence-part already
    # makes "current" unambiguous -- but the metrics table is append-only, so
    # without a pointer, "current" would only ever mean "whichever recipe ran
    # most recently," silently, with no record that a name's meaning moved at
    # all. See resolve_recipe_currency/commit_recipe_currency.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS current_recipes (
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            part TEXT NOT NULL,
            recipe_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (kind, name, part)
        )
        """
    )
    connection.commit()
    return connection


def has_database(project_path):
    """
    Whether this project has a runs_and_metrics.sqlite yet.

    What a READ checks before opening one: `connect` creates the file, and a
    project that has never run anything should stay as it is when something
    merely asks what it holds (see project.paths' "creates nothing").

    - `project_path` -- project to check.
    """
    return paths.runs_and_metrics_path(project_path).exists()


@contextmanager
def open_database(project_path):
    """
    A project's runs_and_metrics.sqlite, with both tables guaranteed to exist, CLOSED on exit.

    A context manager rather than a bare connection: left open, every read a
    summary or an export makes holds a file handle for the rest of the process,
    which on Windows is enough to stop the project directory being moved or
    deleted.

    The schema is ensured on every open rather than once per path. The DDL is
    idempotent and cheap, and remembering which paths are ready is wrong as
    soon as one is deleted and recreated under the same name.

    - `project_path` -- project whose database to open.
    """
    connection = connect(paths.runs_and_metrics_path(project_path))
    try:
        ensure_schema(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


def start_run(project_path, recipe, subset=None, context=None):
    """
    Open a run record for a recipe about to execute, and return its run_id.

    The full recipe spec is written now rather than at the end, so an
    interrupted run still records what it was trying to do.

    - `project_path` -- project to record the run in.
    - `recipe` -- the Recipe being executed (see critterframe.recipes).
    - `subset` -- name of the subset being processed, if the run was scoped to
      one. Recorded but deliberately NOT part of the recipe hash: which
      occurrences a recipe ran over is a property of the run, not of the
      recipe, so processing the rest of a project later continues the same work
      instead of counting as different work.
    - `context` -- JSON-serializable record of what this run covered and what
      its operations fit: the occurrence set as a count and an ids_digest, the
      limit, and any prepare() records. Not hashed either, for the same reason
      as subset.
    """
    if recipe.kind not in RUN_KINDS:
        raise ValueError(
            f"run kind must be one of {RUN_KINDS}, got {recipe.kind!r}"
        )

    with open_database(project_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                kind, name, part, subset, recipe_hash, recipe_json,
                status, context_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe.kind,
                recipe.name,
                recipe.part,
                subset,
                recipe.hash,
                canonical_json(recipe.spec()),
                STATUS_RUNNING,
                None if context is None else canonical_json(context),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        run_id = cursor.lastrowid

    logger.info("started %s run '%s' (run_id=%d, part=%s, recipe=%s)",
                recipe.kind, recipe.name, run_id, recipe.part, recipe.hash)
    return run_id


def finish_run(project_path, run_id, processed=0, skipped=0, failed=0,
               status=STATUS_COMPLETE, flags=None):
    """
    Close a run record with its counts.

    - `processed` -- occurrence-parts this run actually derived something for.
    - `skipped` -- occurrence-parts already covered by an equivalent recipe, so
      no work was repeated.
    - `failed` -- occurrence-parts that raised. Individual failures never stop a
      run; they're counted here and logged as they happen.
    - `status` -- STATUS_COMPLETE, or STATUS_FAILED if the run itself (not an
      individual occurrence) blew up.
    - `flags` -- `{flag: count}` of the operations that called their own result
      doubtful (see `drivers.FLAG_KEYS`), folded into the run's context. An
      operation reports these in its `info`, and until they are recorded here
      they exist only as text on a sampled panel.
    """
    with open_database(project_path) as connection:
        if flags:
            stored = connection.execute(
                "SELECT context_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            context = load_json(stored["context_json"]) or {}
            context["flags"] = flags
            connection.execute("UPDATE runs SET context_json = ? WHERE run_id = ?",
                               (canonical_json(context), run_id))

        connection.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, n_processed = ?, n_skipped = ?,
                n_failed = ?
            WHERE run_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(),
             processed, skipped, failed, run_id),
        )
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    logger.info("finished run %d: processed=%d skipped=%d failed=%d (%s)",
                run_id, processed, skipped, failed, status)
    _append_run_log(project_path, dict(row))
    return run_id


def _append_run_log(project_path, row):
    """
    Append one finished run to runs.jsonl, mirroring export.py's
    exports.jsonl: a human-readable log a person can grep/jq/git-diff without
    opening runs_and_metrics.sqlite.

    Only finish_run calls this -- a run left at STATUS_RUNNING by an
    interrupted process keeps its sqlite row but never reaches the log, the
    same "one line per call that actually completed" reasoning
    paths.imports_log_path's docstring gives for imports.jsonl. This is a
    derived mirror, not a second source of truth: load_runs() from sqlite
    stays the primary, filterable reader.
    """
    record = dict(row)
    record["recipe"] = load_json(record.pop("recipe_json"))
    record["context"] = load_json(record.pop("context_json", None))

    append_jsonl(paths.runs_log_path(project_path), record)


def _seeded_current_hash(connection, kind, name, part):
    """
    The recipe presently designated for (kind, name, part): the pointer if one
    has been written, else whichever hash run history's own insertion order
    would already call current.

    The fallback is what makes turning this on safe for a project that already
    has runs: the first metric run after upgrading seeds the pointer from
    history instead of arbitrarily adopting whatever recipe happens to run
    next, so nothing moves under a name that hasn't actually changed.
    """
    row = connection.execute(
        "SELECT recipe_hash FROM current_recipes WHERE kind = ? AND name = ? AND part = ?",
        (kind, name, part),
    ).fetchone()
    if row is not None:
        return row["recipe_hash"]

    row = connection.execute(
        "SELECT recipe_hash FROM runs WHERE kind = ? AND name = ? AND part = ? "
        "ORDER BY run_id DESC LIMIT 1",
        (kind, name, part),
    ).fetchone()
    return None if row is None else row["recipe_hash"]


def _write_current_recipe(connection, kind, name, part, recipe_hash):
    connection.execute(
        """
        INSERT INTO current_recipes (kind, name, part, recipe_hash, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (kind, name, part) DO UPDATE SET
            recipe_hash = excluded.recipe_hash,
            updated_at = excluded.updated_at
        """,
        (kind, name, part, recipe_hash, datetime.now(timezone.utc).isoformat()),
    )


def resolve_recipe_currency(project_path, kind, name, part, recipe_hash, force):
    """
    Settle whether (kind, name, part) may proceed under recipe_hash, and
    whether the caller still owes commit_recipe_currency() once real work
    confirms the change.

    A no-op for anything but "metric": masks.parquet upserts to a single
    current row per occurrence-part, so a segment run_name pointing at several
    recipe hashes over a project's life is neither ambiguous nor new -- it's
    exactly what resegmenting is. Metrics have no such row to make "current"
    unambiguous on their own; the metrics table is append-only, so without
    this, "current" only ever meant "whichever recipe ran most recently,"
    silently, and export.metrics_wide / records.metrics.latest_values key on
    run_name alone -- two recipes sharing a name would interleave under one
    export column or one group-metric fit with no record they ever differed.

    Raises when (kind, name, part) already points at a different recipe and
    force is falsy, naming both hashes. force=True acknowledges the change
    explicitly instead of it happening as a side effect of write order.

    Returns True when the caller must call commit_recipe_currency() itself
    once it knows real work was done; False when there is nothing left to do
    (first use, or already pointing here) because that case is safe to write
    immediately, before any occurrence is processed.
    """
    if kind != "metric":
        return False

    with open_database(project_path) as connection:
        current = _seeded_current_hash(connection, kind, name, part)
        if current is None or current == recipe_hash:
            _write_current_recipe(connection, kind, name, part, recipe_hash)
            return False

        if not force:
            raise ValueError(
                f"run_name {name!r} currently points at a different metric "
                f"recipe for part {part!r} (current hash {current!r}, this "
                f"run's hash is {recipe_hash!r}). export.metrics_wide and "
                f"records.metrics.latest_values key on run_name alone, so "
                f"proceeding would silently interleave two recipes under one "
                f"name. Pass force=True to move {name!r} onto this recipe -- "
                f"values already on record stay there but stop being current "
                f"-- or give this recipe its own name instead."
            )
        return True


def commit_recipe_currency(project_path, kind, name, part, recipe_hash):
    """
    Move (kind, name, part)'s pointer to recipe_hash.

    Called once a forced recipe change has actually produced at least one
    value, not from resolve_recipe_currency itself: writing the pointer before
    confirming that would mean a run that starts under a forced change and
    then fails for every occurrence empties out every occurrence's current
    value for this name, rather than leaving the previous recipe's values
    (the safer outcome) in place.
    """
    if kind != "metric":
        return
    with open_database(project_path) as connection:
        _write_current_recipe(connection, kind, name, part, recipe_hash)
    logger.info("run_name %r for part %r now points at recipe %s",
               name, part, recipe_hash)


def current_recipe_pointers(project_path, kind="metric"):
    """{(name, part): recipe_hash} for every name with a recorded pointer."""
    if not has_database(project_path):
        return {}
    with open_database(project_path) as connection:
        rows = connection.execute(
            "SELECT name, part, recipe_hash FROM current_recipes WHERE kind = ?",
            (kind,),
        ).fetchall()
    return {(row["name"], row["part"]): row["recipe_hash"] for row in rows}


def current_recipe(project_path, name, part, kind="metric"):
    """
    `(recipe_hash, recipe spec)` currently designated for (kind, name, part), or `(None, None)`.

    Read-only: the pointer where one is written, else the newest run's recipe,
    the same fallback `resolve_recipe_currency` seeds from.

    - `name` -- run name.
    - `part` -- part it measured.
    - `kind` -- run kind.
    """
    if not has_database(project_path):
        return None, None
    with open_database(project_path) as connection:
        recipe_hash = _seeded_current_hash(connection, kind, name, part)
        if recipe_hash is None:
            return None, None
        row = connection.execute(
            "SELECT recipe_json FROM runs WHERE kind = ? AND recipe_hash = ? "
            "ORDER BY run_id DESC LIMIT 1",
            (kind, recipe_hash),
        ).fetchone()
    return recipe_hash, None if row is None else load_json(row["recipe_json"])


def _empty_runs_frame():
    """
    A run table with no rows but every column, so a caller can filter or read a
    column off a project that has never run anything without special-casing it.
    """
    return pd.DataFrame(columns=[
        "run_id", "kind", "name", "part", "subset", "recipe_hash", "status",
        "created_at", "finished_at", "n_processed", "n_skipped", "n_failed",
        "recipe", "context",
    ])


def load_runs(project_path, kind=None, name=None, recipe_hash=None, run_id=None):
    """
    Read run records as a DataFrame, newest first, with the stored recipe spec
    and run context parsed back into `recipe` and `context` columns of dicts.

    - `kind` -- optional "segment"/"metric" filter.
    - `name` -- optional run-name filter.
    - `recipe_hash` -- optional exact-recipe filter, for "when has this exact
      recipe been run before".
    - `run_id` -- optional exact-run filter, for looking up one run by id.
    """
    query = "SELECT * FROM runs"
    conditions = []
    parameters = []
    for column, value in (("kind", kind), ("name", name),
                          ("recipe_hash", recipe_hash), ("run_id", run_id)):
        if value is not None:
            conditions.append(f"{column} = ?")
            parameters.append(value)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, run_id DESC"

    if not has_database(project_path):
        return _empty_runs_frame()

    with open_database(project_path) as connection:
        rows = [dict(row) for row in connection.execute(query, parameters)]

    for row in rows:
        row["recipe"] = load_json(row.pop("recipe_json"))
        row["context"] = load_json(row.pop("context_json", None))

    return pd.DataFrame(rows)
