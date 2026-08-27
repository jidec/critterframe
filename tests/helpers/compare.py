"""
Making two results comparable without making the comparison meaningless.

Every table CritterFrame writes carries a timestamp, and nothing in the package
ever reads one: masks go stale by derivation hash, calibrations resolve by scope
breadth, runs order by `run_id`, and a registered model's identity is its
fingerprint. Timestamps are provenance for a human, so a test strips them and
compares everything else exactly, rather than comparing loosely and letting a
real difference through.

The one place time is behaviour rather than provenance -- the `_1` suffix on an
import archived twice in one day -- is tested by freezing the clock, not by
stripping anything. See `tests/unit/test_ingest.py`.
"""

import re

# Written by records.masks, records.calibrations, records.runs, records.models,
# and training.datasets. `run_id`/`metric_id` are sqlite insertion order: real
# and assertable on their own, but not part of a value comparison.
VOLATILE_COLUMNS = (
    "created_at",
    "finished_at",
    "registered_at",
    "run_created_at",
    "metric_id",
    "run_id",
    "source_mtime",
    "source_path",
)

ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")


def strip_volatile(df, extra=()):
    """
    A copy of `df` without the columns that move between two identical runs,
    reindexed so row order is the only thing left to disagree about.

    extra -- further columns to drop for one particular comparison.
    """
    columns = [column for column in tuple(VOLATILE_COLUMNS) + tuple(extra)
               if column in df.columns]
    return df.drop(columns=columns).reset_index(drop=True)


def is_iso_utc(value):
    """
    Whether a timestamp is ISO-8601 in UTC -- asserted once per writer, so that
    stripping the column everywhere else doesn't mean nobody ever checks it.
    """
    return bool(ISO_UTC.match(str(value)))
