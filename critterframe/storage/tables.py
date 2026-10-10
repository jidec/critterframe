"""Parquet tables: read, snapshot write, upsert."""

import logging
import os
import uuid
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _atomic_to_parquet(df, table_path):
    """Write `df` through a temp file, so an interrupted write never leaves a truncated table."""
    table_path = Path(table_path)
    tmp_path = table_path.with_name(f"{table_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, table_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_table(new_df, table_path):
    """Write a parquet table, replacing whatever was there.

    Args:
        new_df: The full table.
        table_path: Destination parquet path.
    """
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    out = new_df.reset_index(drop=True)
    _atomic_to_parquet(out, table_path)
    logger.info("wrote table -> %s (%d rows)", table_path, len(out))
    return out


def _check_keys(df, key_cols, what):
    """Raise unless every key column is present and free of nulls."""
    missing = [column for column in key_cols if column not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing key column(s) {missing} (columns: {sorted(df.columns)})")

    null_counts = {column: int(df[column].isna().sum()) for column in key_cols if df[column].isna().any()}
    if null_counts:
        raise ValueError(
            f"{what} has null values in key column(s) {null_counts} -- a key must identify a row"
        )


def upsert_table(new_df, table_path, key_cols):
    """Merge rows into a parquet table, replacing rows that match on the key and appending the rest.

    Args:
        new_df: Rows to merge in; they win on conflict.
        table_path: Destination parquet path.
        key_cols: Columns identifying a row, e.g. `["occurrence_id", "part"]`. Compared by
            value with no coercion, and must be non-null.
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
            combined = pd.concat([existing[~existing_keys.isin(new_keys)], new_df], ignore_index=True)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.reset_index(drop=True)
    _atomic_to_parquet(combined, table_path)
    logger.info("upserted %d rows -> %s (%d total)", len(new_df), table_path, len(combined))
    return combined


def load_table(table_path, columns=None, missing_ok=False, filters=None):
    """Read a parquet table.

    Args:
        table_path: Parquet path.
        columns: Column names to read; all if None.
        missing_ok: Return an empty DataFrame for a missing file instead of raising.
        filters: `[(column, op, value), ...]` rows must all satisfy, applied while the file
            is scanned. A filter column need not be among `columns`.
    """
    if not Path(table_path).exists():
        if missing_ok:
            return pd.DataFrame(columns=list(columns) if columns else [])
        raise FileNotFoundError(f"no table at {table_path}. Ingest or run the step that produces it first.")
    return pd.read_parquet(table_path, columns=columns, filters=filters or None)


def table_columns(table_path):
    """Return the column names a parquet holds, read from its footer; empty if the file is missing."""
    table_path = Path(table_path)
    if not table_path.exists():
        return []
    import pyarrow.parquet

    return list(pyarrow.parquet.read_schema(table_path).names)
