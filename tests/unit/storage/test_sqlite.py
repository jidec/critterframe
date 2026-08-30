"""
Opening a project's sqlite database.

Mechanics only -- the runs/metrics schema built on top of this lives in
records.runs, and is tested there.

The pragmas are the load-bearing part and are asserted rather than assumed:
WAL plus a busy_timeout are what let several processes write to one project's
runs_and_metrics.sqlite without a writer raising `database is locked` on first
contention, which is what a sharded segmentation run does by design (see
tests/integration/test_sharded_segmentation.py). Foreign keys are off by
default in sqlite, and without them an orphaned metric row -- a value whose run
record is gone -- is accepted silently.
"""

from critterframe.storage.sqlite import BUSY_TIMEOUT_MS, connect


def test_connect_creates_the_parent_and_returns_rows_by_name(tmp_path):
    database = tmp_path / "project" / "runs.sqlite"
    with connect(database) as connection:
        connection.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        connection.execute("INSERT INTO t VALUES (1, 'x')")
        row = connection.execute("SELECT * FROM t").fetchone()

    assert database.exists()
    assert row["a"] == 1 and row["b"] == "x"


def test_foreign_keys_are_enforced(tmp_path):
    """
    Metric rows reference their run. Without the pragma sqlite accepts an
    orphan silently, and an orphaned value is one whose provenance is gone.
    """
    with connect(tmp_path / "runs.sqlite") as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_wal_mode_and_busy_timeout_are_set(tmp_path):
    """
    Needed the moment more than one process writes to the same project (e.g.
    several sharded segmentation runs sharing runs_and_metrics.sqlite):
    without these, a writer that finds the database locked raises instead of
    waiting.
    """
    with connect(tmp_path / "runs.sqlite") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
