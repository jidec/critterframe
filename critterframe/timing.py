"""
timed(): one "<label> in Ns" log line around a slow step.

A multi-gigabyte ingest spends minutes inside single calls, and a caller
watching the log otherwise sees nothing between "start" and "done".
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed(label, log=None, **counts):
    """
    Log how long the block took, once it finishes.

    Nothing is logged if the block raises: a line saying a step finished, when
    it didn't, is worse than no line. What's worth reporting is usually only
    known once the work is done, so the block gets a dict to fill in --
    `with timed("transformed") as done: ...; done["rows"] = len(df)` prints
    `transformed in 1.2s (rows=8000)`.

    - `label` -- what finished, in the past tense, e.g. `"archived"`.
    - `log` -- the logging callable to use, so a line carries its own
      module's name. Defaults to this module's `logger.info`.
    - `counts` -- initial entries of the yielded dict, for what's already
      known going in.
    """
    start = time.monotonic()
    counts = dict(counts)
    yield counts

    detail = ", ".join(f"{key}={value}" for key, value in counts.items())
    (log or logger.info)("%s in %.1fs%s", label, time.monotonic() - start,
                         f" ({detail})" if detail else "")
