"""
Run records: one execution of a recipe over a particular set of occurrences.

Runs are of segmentation and metrics only, because these derive something and need run recorded for provenance

Ingest and download aren't runs: they bring data in rather than deriving anything from it, and their provenance
is the immutable import they archived.

Runs live in project_path/runs_and_metrics.sqlite alongside the metric values
they produced (see records.metrics). sqlite rather than parquet because a run
appends row by row over a long, interruptible process and has to survive the
process dying partway through -- a rewrite-the-whole-file format would either
lose everything or need constant rewriting.

A run row is opened when the run starts and closed when it ends, so a run
interrupted halfway leaves a "running" row rather than disappearing -- which is
how you tell "this never finished" apart from "this finished and found nothing
to do".
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..recipes import canonical_json, load_json
from ..storage.tables import connect

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
            created_at TEXT NOT NULL,
            finished_at TEXT,
            n_processed INTEGER NOT NULL DEFAULT 0,
            n_skipped INTEGER NOT NULL DEFAULT 0,
            n_failed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
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
    connection.commit()
    return connection


def open_database(project_path):
    """Open a project's runs_and_metrics.sqlite with both tables guaranteed to exist."""
    return ensure_schema(connect(paths.runs_and_metrics_path(project_path)))


def start_run(project_path, recipe, subset=None):
    """
    Open a run record for a recipe about to execute, and return its run_id.

    The full recipe spec is written now rather than at the end, so an
    interrupted run still records what it was trying to do.

    project_path -- project to record the run in.
    recipe       -- the Recipe being executed (see critterframe.recipes).
    subset       -- name of the subset being processed, if the run was scoped
                    to one. Recorded but deliberately NOT part of the recipe
                    hash: which occurrences a recipe ran over is a property of
                    the run, not of the recipe, so processing the rest of a
                    project later continues the same work instead of counting
                    as different work.
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
                status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe.kind,
                recipe.name,
                recipe.part,
                subset,
                recipe.hash,
                canonical_json(recipe.spec()),
                STATUS_RUNNING,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        run_id = cursor.lastrowid

    logger.info("started %s run '%s' (run_id=%d, part=%s, recipe=%s)",
                recipe.kind, recipe.name, run_id, recipe.part, recipe.hash)
    return run_id


def finish_run(project_path, run_id, processed=0, skipped=0, failed=0,
               status=STATUS_COMPLETE):
    """
    Close a run record with its counts.

    processed -- occurrence-parts this run actually derived something for.
    skipped   -- occurrence-parts already covered by an equivalent recipe, so
                 no work was repeated.
    failed    -- occurrence-parts that raised. Individual failures never stop a
                 run; they're counted here and logged as they happen.
    status    -- STATUS_COMPLETE, or STATUS_FAILED if the run itself (not an
                 individual occurrence) blew up.
    """
    with open_database(project_path) as connection:
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

    logger.info("finished run %d: processed=%d skipped=%d failed=%d (%s)",
                run_id, processed, skipped, failed, status)
    return run_id


def load_runs(project_path, kind=None, name=None, recipe_hash=None):
    """
    Read run records as a DataFrame, newest first, with the stored recipe spec
    parsed back into a `recipe` column of dicts.

    kind        -- optional "segment"/"metric" filter.
    name        -- optional run-name filter.
    recipe_hash -- optional exact-recipe filter, for "when has this exact
                   recipe been run before".
    """
    query = "SELECT * FROM runs"
    conditions = []
    parameters = []
    for column, value in (("kind", kind), ("name", name),
                          ("recipe_hash", recipe_hash)):
        if value is not None:
            conditions.append(f"{column} = ?")
            parameters.append(value)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, run_id DESC"

    with open_database(project_path) as connection:
        rows = [dict(row) for row in connection.execute(query, parameters)]

    for row in rows:
        row["recipe"] = load_json(row.pop("recipe_json"))

    return pd.DataFrame(rows)
