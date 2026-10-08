"""The failures ledger: what failed and under which attempt, so an unchanged repeat is not retried."""

import logging
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..storage.tables import load_table, upsert_table, write_table

logger = logging.getLogger(__name__)

KEY_COLS = ["occurrence_id", "part", "stage"]

COLUMNS = ["occurrence_id", "part", "stage", "context_hash", "error", "failed_at"]

# part for a stage with no part concept, e.g. download.
NO_PART = ""

# Errors recorded before a missing input counted as no_input rather than a
# failure. The context_hash never changes when the image finally arrives, so
# honouring these rows would skip the occurrence forever.
NOT_FAILURES = {"no image in the image store"}


def record_failures(project_path, stage, rows):
    """Record one failed attempt per row, replacing any earlier one for the same occurrence-part.

    Args:
        project_path: Project to write to.
        stage: What kind of attempt failed, e.g. `"download"` or `"segment"`.
        rows: `{"occurrence_id", "part", "context_hash", "error"}` dicts. `part` is optional;
            `context_hash` identifies what was attempted.
    """
    if not rows:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "occurrence_id": str(row["occurrence_id"]),
            "part": str(row.get("part", NO_PART) or NO_PART),
            "stage": stage,
            "context_hash": row["context_hash"],
            "error": str(row["error"]),
            "failed_at": now,
        }
        for row in rows
    ]
    upsert_table(pd.DataFrame(records, columns=COLUMNS), paths.failures_path(project_path), key_cols=KEY_COLS)
    return len(records)


def failed_keys(project_path, stage, context_hashes):
    """Return the occurrence-parts recorded as failed under the attempt about to be made.

    A stored failure whose context hash differs is not returned, so it is attempted again.

    Args:
        project_path: Project to read from.
        stage: Stage to look in.
        context_hashes: `{(occurrence_id, part): context_hash}` of what is about to be attempted.

    Returns:
        A set of `(occurrence_id, part)`.
    """
    if not context_hashes:
        return set()

    df = load_table(paths.failures_path(project_path), missing_ok=True)
    df = df[df["stage"] == stage] if "stage" in df.columns and not df.empty else df
    if not df.empty:
        df = df[~df["error"].isin(NOT_FAILURES)]
    if df.empty:
        return set()

    stored = {(row.occurrence_id, row.part): row.context_hash for row in df.itertuples(index=False)}
    return {key for key, context_hash in context_hashes.items() if stored.get(key) == context_hash}


def clear_failures(project_path, stage, keys):
    """Drop the recorded failures of keys that have since succeeded.

    Args:
        project_path: Project to write to.
        stage: Stage the keys belong to.
        keys: Iterable of `(occurrence_id, part)`.
    """
    keys = {(str(occurrence_id), str(part)) for occurrence_id, part in keys}
    if not keys:
        return 0

    path = paths.failures_path(project_path)
    df = load_table(path, missing_ok=True)
    if df.empty:
        return 0
    df = df.reset_index(drop=True)

    stage_mask = df["stage"] == stage
    key_mask = pd.Series(list(zip(df["occurrence_id"], df["part"]))).isin(keys)
    to_drop = stage_mask & key_mask
    if not to_drop.any():
        return 0

    write_table(df[~to_drop], path)
    return int(to_drop.sum())


def load_failures(project_path, stage=None, columns=None):
    """Read the failures ledger.

    Args:
        project_path: Project to read from.
        stage: Stage to narrow to; None for every stage.
        columns: Columns to read; all if None.
    """
    df = load_table(paths.failures_path(project_path), columns=columns, missing_ok=True)
    if stage is not None and not df.empty and "stage" in df.columns:
        df = df[df["stage"] == stage].reset_index(drop=True)
    return df
