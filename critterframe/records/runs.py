"""Run records: the sqlite schema for runs, metrics and the current-recipe pointer, with its migrations."""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..recipes import canonical_json, describe_recipe_change, load_json
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
    """Add any `runs` column introduced after this database was created."""
    stored = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
    for column, declaration in ADDED_RUN_COLUMNS.items():
        if column not in stored:
            logger.info("migrating runs table: adding '%s' column", column)
            connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {declaration}")


def _drop_legacy_metric_columns(connection):
    """Drop the per-row `version` and `created_at` columns an older metrics table carries.

    The old `created_at` was NOT NULL, so a table still holding it rejects every new insert.
    """
    stored = {row["name"] for row in connection.execute("PRAGMA table_info(metrics)")}
    for column in LEGACY_METRIC_COLUMNS:
        if column in stored:
            logger.info("migrating metrics table: dropping legacy '%s' column", column)
            connection.execute(f"ALTER TABLE metrics DROP COLUMN {column}")


def ensure_schema(connection):
    """Create the runs, metrics and current-recipe tables if missing, and migrate older ones."""
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
    """Return whether the project has a runs and metrics database yet.

    Args:
        project_path: Project to check.
    """
    return paths.runs_and_metrics_path(project_path).exists()


@contextmanager
def open_database(project_path):
    """Open a project's runs and metrics database as a context manager that closes on exit.

    Args:
        project_path: Project whose database to open; created if missing.
    """
    connection = connect(paths.runs_and_metrics_path(project_path))
    try:
        ensure_schema(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


def start_run(project_path, recipe, subset=None, context=None):
    """Open a run record for a recipe about to execute, and return its run id.

    Args:
        project_path: Project to record the run in.
        recipe: The `Recipe` being executed.
        subset: Name of the subset being processed. Recorded, not hashed.
        context: JSON-serializable record of what the run covers and what its operations
            fit. Recorded, not hashed.
    """
    if recipe.kind not in RUN_KINDS:
        raise ValueError(f"run kind must be one of {RUN_KINDS}, got {recipe.kind!r}")

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

    logger.info(
        "started %s run '%s' (run_id=%d, part=%s, recipe=%s)",
        recipe.kind,
        recipe.name,
        run_id,
        recipe.part,
        recipe.hash,
    )
    return run_id


def finish_run(project_path, run_id, processed=0, skipped=0, failed=0, status=STATUS_COMPLETE, flags=None):
    """Close a run record with its counts, and append it to `runs.jsonl`.

    Args:
        project_path: Project the run belongs to.
        run_id: The run.
        processed: Occurrence-parts the run derived something for.
        skipped: Occurrence-parts already covered by an equivalent recipe.
        failed: Occurrence-parts that raised.
        status: `STATUS_COMPLETE`, or `STATUS_FAILED` if the run itself failed.
        flags: `{flag: count}` of results their own operations called doubtful, folded
            into the run's context.
    """
    with open_database(project_path) as connection:
        if flags:
            stored = connection.execute(
                "SELECT context_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            context = load_json(stored["context_json"]) or {}
            context["flags"] = flags
            connection.execute(
                "UPDATE runs SET context_json = ? WHERE run_id = ?", (canonical_json(context), run_id)
            )

        connection.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, n_processed = ?, n_skipped = ?,
                n_failed = ?
            WHERE run_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), processed, skipped, failed, run_id),
        )
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    logger.info(
        "finished run %d: processed=%d skipped=%d failed=%d (%s)", run_id, processed, skipped, failed, status
    )
    _append_run_log(project_path, dict(row))
    return run_id


def _append_run_log(project_path, row):
    """Append one finished run to `runs.jsonl`, the readable mirror of the runs table."""
    record = dict(row)
    record["recipe"] = load_json(record.pop("recipe_json"))
    record["context"] = load_json(record.pop("context_json", None))

    append_jsonl(paths.runs_log_path(project_path), record)


def _seeded_current_hash(connection, kind, name, part):
    """Return the recipe hash `(kind, name, part)` points at.

    The pointer where one is written, else the hash run history's insertion order makes
    newest, so a project with runs from before the pointer existed keeps what it had.
    """
    row = connection.execute(
        "SELECT recipe_hash FROM current_recipes WHERE kind = ? AND name = ? AND part = ?",
        (kind, name, part),
    ).fetchone()
    if row is not None:
        return row["recipe_hash"]

    row = connection.execute(
        "SELECT recipe_hash FROM runs WHERE kind = ? AND name = ? AND part = ? ORDER BY run_id DESC LIMIT 1",
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


def resolve_recipe_currency(project_path, kind, name, part, recipe_hash, force, recipe_spec=None):
    """Decide whether a metric run name may proceed under a recipe hash.

    Does nothing for a segment run.

    Args:
        project_path: Project the run belongs to.
        kind: Run kind.
        name: Run name.
        part: Part being measured.
        recipe_hash: Hash of the recipe about to run.
        force: Allow the name to move onto a different recipe.
        recipe_spec: This run's `Recipe.spec()`, so the error can say what differs.

    Returns:
        True when the caller must call `commit_recipe_currency` once a value is stored;
        False when the pointer already points here or was written now.

    Raises:
        ValueError: If the name points at a different recipe and `force` is falsy.
    """
    if kind != "metric":
        return False

    with open_database(project_path) as connection:
        current = _seeded_current_hash(connection, kind, name, part)
        if current is None or current == recipe_hash:
            _write_current_recipe(connection, kind, name, part, recipe_hash)
            return False

        differs = ""
        if recipe_spec is not None:
            row = connection.execute(
                "SELECT recipe_json FROM runs WHERE kind = ? AND recipe_hash = ? "
                "ORDER BY run_id DESC LIMIT 1",
                (kind, current),
            ).fetchone()
            if row is not None:
                changed = describe_recipe_change(load_json(row["recipe_json"]), recipe_spec)
                differs = "\nWhat this run changes from the recipe the name points at:\n" + "\n".join(
                    f"  - {line}" for line in changed
                )

        if not force:
            raise ValueError(
                f"run_name {name!r} currently points at a different metric "
                f"recipe for part {part!r} (current hash {current!r}, this "
                f"run's hash is {recipe_hash!r}). export.metrics_wide and "
                f"records.metrics.latest_values key on run_name alone, so "
                f"proceeding would silently interleave two recipes under one "
                f"name. Pass force=True to move {name!r} onto this recipe -- "
                f"values already on record stay there but stop being current "
                f"-- or give this recipe its own name instead.{differs}"
            )
        if differs:
            logger.info(
                "run_name %r part %r: force=True moves it from recipe %s to %s.%s",
                name,
                part,
                current,
                recipe_hash,
                differs,
            )
        return True


def commit_recipe_currency(project_path, kind, name, part, recipe_hash):
    """Move a run name's pointer to a recipe hash.

    Args:
        project_path: Project the run belongs to.
        kind: Run kind.
        name: Run name.
        part: Part measured.
        recipe_hash: Hash to point at.
    """
    if kind != "metric":
        return
    with open_database(project_path) as connection:
        _write_current_recipe(connection, kind, name, part, recipe_hash)
    logger.info("run_name %r for part %r now points at recipe %s", name, part, recipe_hash)


def current_recipe_pointers(project_path, kind="metric"):
    """Return `{(name, part): recipe_hash}` for every name with a recorded pointer."""
    if not has_database(project_path):
        return {}
    with open_database(project_path) as connection:
        rows = connection.execute(
            "SELECT name, part, recipe_hash FROM current_recipes WHERE kind = ?",
            (kind,),
        ).fetchall()
    return {(row["name"], row["part"]): row["recipe_hash"] for row in rows}


def current_recipe(project_path, name, part, kind="metric"):
    """Return the `(recipe_hash, recipe spec)` a run name points at, or `(None, None)`.

    Args:
        project_path: Project to read from.
        name: Run name.
        part: Part it measured.
        kind: Run kind.
    """
    if not has_database(project_path):
        return None, None
    with open_database(project_path) as connection:
        recipe_hash = _seeded_current_hash(connection, kind, name, part)
        if recipe_hash is None:
            return None, None
        row = connection.execute(
            "SELECT recipe_json FROM runs WHERE kind = ? AND recipe_hash = ? ORDER BY run_id DESC LIMIT 1",
            (kind, recipe_hash),
        ).fetchone()
    return recipe_hash, None if row is None else load_json(row["recipe_json"])


def _empty_runs_frame():
    """Return a run table with every column and no rows."""
    return pd.DataFrame(
        columns=[
            "run_id",
            "kind",
            "name",
            "part",
            "subset",
            "recipe_hash",
            "status",
            "created_at",
            "finished_at",
            "n_processed",
            "n_skipped",
            "n_failed",
            "recipe",
            "context",
        ]
    )


def load_runs(project_path, kind=None, name=None, recipe_hash=None, run_id=None):
    """Read run records as a DataFrame, newest first.

    The stored recipe and context are parsed into `recipe` and `context` columns of dicts.

    Args:
        project_path: Project to read from.
        kind: `"segment"` or `"metric"`.
        name: Run name.
        recipe_hash: Exact recipe.
        run_id: Exact run.
    """
    query = "SELECT * FROM runs"
    conditions = []
    parameters = []
    for column, value in (("kind", kind), ("name", name), ("recipe_hash", recipe_hash), ("run_id", run_id)):
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
