"""
The sqlite connection for a project's runs & metrics database.

Mechanics only; the schema belongs to records.runs. sqlite is used here because
rows are appended one at a time over a long run that must survive a process
dying halfway through.
"""

import logging
import sqlite3
import time
from pathlib import Path

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


def connect(database_path):
    """
    Open a sqlite database, creating its parent directory if needed, with row
    access by column name.

    Callers create their own tables; this only opens the file. WAL mode and
    busy_timeout are what let several sharded runs write to one project at
    once (see CLAUDE.md).
    """
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    # busy_timeout must be set BEFORE this, and this needs its own retry: the
    # one-time switch into WAL raises SQLITE_LOCKED, which busy_timeout doesn't
    # cover (it only retries SQLITE_BUSY). Fails only under concurrency, on a
    # brand-new project.
    for attempt in range(WAL_SWITCH_RETRIES):
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            break
        except sqlite3.OperationalError:
            if attempt == WAL_SWITCH_RETRIES - 1:
                raise
            time.sleep(WAL_SWITCH_RETRY_DELAY_S)

    return connection
