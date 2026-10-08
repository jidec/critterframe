"""timed(): one "<label> in Ns" log line around a slow step."""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed(label, log=None, **counts):
    """Log how long the block took, once it finishes; nothing if it raises.

    Yields a dict the block can add counts to, which are appended to the line:
    `with timed("transformed") as done: ...; done["rows"] = len(df)` logs
    `transformed in 1.2s (rows=8000)`.

    Args:
        label: What finished, in the past tense, e.g. `"archived"`.
        log: Logging callable, so a line carries the caller's module name.
        **counts: Initial entries of the yielded dict.
    """
    start = time.monotonic()
    counts = dict(counts)
    yield counts

    detail = ", ".join(f"{key}={value}" for key, value in counts.items())
    (log or logger.info)("%s in %.1fs%s", label, time.monotonic() - start, f" ({detail})" if detail else "")
