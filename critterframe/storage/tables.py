"""
Generic parquet/sqlite storage helpers with no knowledge of a specific
entity or table

Records/ modules supply the schema, this supplies the mechanics.

Two storage shapes live here because CritterFrame uses each for what it's
good at:

  parquet -- occurrences and masks: wide, columnar, read whole or by column,
             rewritten as a unit.
  sqlite  -- runs and metric values: appended row by row over long
             interruptible runs, queried by key, and needing to survive a
             process dying halfway through.

Writing a table comes in two shapes, and which one is right is a property of
the data, not of the caller. write_table replaces wholesale, for tables fed by
full snapshots -- external occurrence exports are snapshots rather than deltas,
so the newest one IS the complete desired state, and the dated import ingest
archived is the recovery path if a replacement is ever wrong. upsert_table
merges on a key, for tables built up incrementally across many runs.
"""

import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# How long a writer waits for a lock before giving up, once more than one
# process is writing to the same project's runs_and_metrics.sqlite (e.g.
# several sharded segmentation runs sharing one project -- see
# segmentation.run.run_segments' shard= parameter). Without this, sqlite's
# default is to raise "database is locked" immediately rather than wait.
BUSY_TIMEOUT_MS = 30_000

# Retry budget for the one-time switch into WAL mode specifically (see
# connect()) -- separate from BUSY_TIMEOUT_MS because that pragma doesn't
# cover this particular race.
WAL_SWITCH_RETRIES = 20
WAL_SWITCH_RETRY_DELAY_S = 0.05


def write_table(new_df, table_path):
    """
    Write a working parquet, overwriting whatever was there.

    new_df     -- the full table to write.
    table_path -- destination parquet path; overwritten wholesale.
    """
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    out = new_df.reset_index(drop=True)
    out.to_parquet(table_path)
    logger.info("wrote table -> %s (%d rows)", table_path, len(out))
    return out


def _check_keys(df, key_cols, what):
    """
    Raise unless every key column is present and free of nulls.

    A null key isn't an identifier -- it can't be matched against, and letting
    one through means a row that silently never upserts and instead accumulates
    a duplicate on every run. Loud here beats a mask table that grows a second
    copy of the same occurrence-part each time it's segmented.
    """
    missing = [column for column in key_cols if column not in df.columns]
    if missing:
        raise KeyError(
            f"{what} is missing key column(s) {missing} "
            f"(columns: {sorted(df.columns)})"
        )

    null_counts = {
        column: int(df[column].isna().sum())
        for column in key_cols
        if df[column].isna().any()
    }
    if null_counts:
        raise ValueError(
            f"{what} has null values in key column(s) {null_counts} -- a key "
            "must identify a row"
        )


def upsert_table(new_df, table_path, key_cols):
    """
    Merge rows into a working parquet, replacing any existing row that matches
    on key_cols and appending the rest.

    Used where a table is built up INCREMENTALLY over many interruptible runs
    rather than snapshotted -- masks especially, where a run over a subset must
    not disturb the masks another subset's recipe produced (a project's subsets
    are allowed to be processed with entirely different recipes, so a
    replace-everything write would destroy the other subsets' work).

    Keys are compared BY VALUE, with no coercion. Coercing them to string
    first would look harmless and quietly do three wrong things: it would make
    the integer 1 and the string "1" the same key while leaving 1 and 1.0
    different; and it would turn None, NaN, and pd.NA into "None", "nan", and
    "<NA>" -- three distinct keys, each of which a literal string in the data
    could then collide with. Guaranteeing key types is the records layer's job
    (see records.masks.make_mask_row); storage compares what it is given and
    rejects what can't be an identifier.

    new_df     -- rows to merge in; wins on conflict.
    table_path -- destination parquet path.
    key_cols   -- columns identifying a row, e.g. ["occurrence_id", "part"].
                  Must be present and non-null in both frames.
    """
    table_path = Path(table_path)
    _check_keys(new_df, key_cols, "rows being written")

    combined = new_df
    if table_path.exists():
        existing = pd.read_parquet(table_path)
        if len(existing):
            _check_keys(existing, key_cols, f"existing table {table_path}")

            # Without coercion, a key column whose type has changed can never
            # match what's already stored -- every row would append instead of
            # replacing, and the parquet write would fail further down with an
            # Arrow type error naming neither the cause nor the fix.
            changed = {
                column: (str(existing[column].dtype), str(new_df[column].dtype))
                for column in key_cols
                if existing[column].dtype != new_df[column].dtype
            }
            if changed:
                raise TypeError(
                    f"key column type(s) changed since {table_path} was "
                    f"written: {changed} (existing, new). Keys are compared by "
                    "value, so these can never match -- fix the type where the "
                    "rows are built, in the records layer"
                )

            existing_keys = pd.MultiIndex.from_frame(existing[key_cols])
            new_keys = pd.MultiIndex.from_frame(new_df[key_cols])
            combined = pd.concat([existing[~existing_keys.isin(new_keys)], new_df],
                                 ignore_index=True)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.reset_index(drop=True)
    combined.to_parquet(table_path)
    logger.info("upserted %d rows -> %s (%d total)",
                len(new_df), table_path, len(combined))
    return combined


def load_table(table_path, columns=None, missing_ok=False):
    """
    Read a working parquet table. Pass columns=[...] to read only some columns
    off disk.

    table_path -- parquet path written by write_table()/upsert_table().
    columns    -- optional list of column names to read; reads all if None.
    missing_ok -- True returns an empty DataFrame (with `columns` as its
                  columns, if given) when the file doesn't exist, for tables
                  that are legitimately absent until something writes them
                  (masks before any segmentation run). False (default) raises,
                  so a missing occurrences table doesn't quietly read as "this
                  project has no occurrences".
    """
    if not Path(table_path).exists():
        if missing_ok:
            return pd.DataFrame(columns=list(columns) if columns else [])
        raise FileNotFoundError(
            f"no table at {table_path}. Ingest or run the step that produces "
            "it first."
        )
    return pd.read_parquet(table_path, columns=columns)


def table_columns(table_path):
    """
    The column names a parquet holds, read from its footer without touching any
    of its data. Empty list if the file doesn't exist.

    Here so a records module can ask what an EXISTING table has before deciding
    what to read from it. Parquet schemas here grow over time -- a column added
    to a record type doesn't exist in a project segmented last month -- and a
    read that names a column the file predates fails outright, so "read these
    columns if they're there" needs an answer that isn't a swallowed exception.
    """
    table_path = Path(table_path)
    if not table_path.exists():
        return []
    import pyarrow.parquet

    return list(pyarrow.parquet.read_schema(table_path).names)


def connect(database_path):
    """
    Open a sqlite database, creating its parent directory if needed, with
    row access by column name.

    Callers are responsible for creating their own tables (see
    records.runs.ensure_schema) -- this only opens the file.

    WAL journal mode lets readers proceed while a writer commits, instead of
    a writer's commit blocking every reader; busy_timeout makes a writer that
    finds the database locked BLOCK AND RETRY (at the sqlite C level) for up
    to BUSY_TIMEOUT_MS rather than raising `database is locked` on first
    contention. Neither matters for a single process talking to its own
    project, but both do the moment two do -- several sharded run_segments()
    calls each doing their own start_run()/finish_run() against one project's
    runs_and_metrics.sqlite, say.

    The one-time switch INTO WAL mode is a separate race busy_timeout does
    NOT cover: sqlite reports it as SQLITE_LOCKED ("database is locked"), not
    SQLITE_BUSY ("database is busy"), and busy_timeout's automatic retry only
    catches the latter. Two connections opening a brand-new project's
    database at nearly the same moment can hit this genuinely, so it gets its
    own short, bounded, Python-level retry -- once any connection succeeds,
    WAL is recorded in the file itself and every later open just sees it
    already set, so this only ever matters in that first narrow window.
    """
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    for attempt in range(WAL_SWITCH_RETRIES):
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            break
        except sqlite3.OperationalError:
            if attempt == WAL_SWITCH_RETRIES - 1:
                raise
            time.sleep(WAL_SWITCH_RETRY_DELAY_S)

    return connection
