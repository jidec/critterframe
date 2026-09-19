"""
A persistent ledger of failed attempts, so a rerun doesn't retry work already
known not to work.

One small parquet table (failures.parquet) shared by any stage that wants this
pattern -- download and segmentation today. A failure is scoped by a
context_hash naming what was attempted, so a change that could plausibly fix
it (a corrected URL from a re-ingest, a retuned recipe) drops the old failure
and is retried automatically, the same way records.masks.completed_keys scopes
a derived mask's completion on its upstream's source_mask_hash.
"""

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
    """
    Persist one failed attempt per row, replacing any prior failure recorded
    for the same occurrence-part under this stage.

    - `project_path` -- the project.
    - `stage` -- what kind of attempt failed, e.g. "download" or "segment".
    - `rows` -- [{"occurrence_id", "part" (optional, NO_PART if omitted),
      "context_hash", "error"}, ...]. context_hash identifies what was
      attempted, so failed_keys() can tell a repeat of the same attempt from a
      changed one.
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
    upsert_table(pd.DataFrame(records, columns=COLUMNS),
                paths.failures_path(project_path), key_cols=KEY_COLS)
    return len(records)


def failed_keys(project_path, stage, context_hashes):
    """
    The (occurrence_id, part) pairs already recorded as failed for this stage
    with the SAME context_hash they'd be attempted under now.

    A stored failure whose context_hash no longer matches -- a corrected URL,
    a retuned recipe -- is not returned, so it is attempted again with no flag
    needed, the same way a changed source_mask_hash drops a mask from
    completed_keys.

    - `context_hashes` -- {(occurrence_id, part): context_hash} of what is
      about to be attempted.
    """
    if not context_hashes:
        return set()

    df = load_table(paths.failures_path(project_path), missing_ok=True)
    df = df[df["stage"] == stage] if "stage" in df.columns and not df.empty else df
    if not df.empty:
        df = df[~df["error"].isin(NOT_FAILURES)]
    if df.empty:
        return set()

    stored = {
        (row.occurrence_id, row.part): row.context_hash
        for row in df.itertuples(index=False)
    }
    return {
        key for key, context_hash in context_hashes.items()
        if stored.get(key) == context_hash
    }


def clear_failures(project_path, stage, keys):
    """
    Drop recorded failures for keys that just succeeded.

    Hygiene, not correctness -- a stale row's context_hash simply never
    matches again -- but cheap to do at the point the caller already knows
    which keys succeeded, so the table doesn't carry rows that can never be
    read as failed again.

    - `keys` -- iterable of (occurrence_id, part).
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
    """
    Read failures.parquet, optionally narrowed to one stage -- for an operator
    to see what's being skipped and why.

    - `stage` -- restrict to one stage, or None for every stage.
    """
    df = load_table(paths.failures_path(project_path), columns=columns, missing_ok=True)
    if stage is not None and not df.empty and "stage" in df.columns:
        df = df[df["stage"] == stage].reset_index(drop=True)
    return df
