"""
Parquet tables (occurrences & masks): read, snapshot write, upsert.

Mechanics only; records/ modules supply the schema. write_table replaces
wholesale, for tables fed by full snapshots; upsert_table merges on a key, for
tables built up across many runs.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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

    A null key can't be matched against, so a row carrying one never upserts --
    it accumulates a duplicate on every run instead of replacing.
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

    For tables built up across many runs, where a run over one subset must not
    disturb another subset's rows. Keys are compared by value with no coercion
    (see CLAUDE.md; the records layer guarantees their types).

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
    missing_ok -- True returns an empty DataFrame for a table that is
                  legitimately absent until something writes it, e.g. masks
                  before any segmentation run. False (the default) raises, so a
                  missing occurrence table isn't read as an empty project.
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

    Schemas grow over time, so a column added to a record type won't exist in a
    project segmented last month. Reading one that isn't there fails outright;
    this is how a caller asks first.
    """
    table_path = Path(table_path)
    if not table_path.exists():
        return []
    import pyarrow.parquet

    return list(pyarrow.parquet.read_schema(table_path).names)
