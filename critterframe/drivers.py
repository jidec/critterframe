"""What every per-item driver shares: Tally (its summary), Progress (its progress line), NoInput."""

import logging
import time
from collections import Counter, deque

logger = logging.getLogger(__name__)

# `info` keys an operation sets to say its own result is doubtful. `degenerate`
# means it returned the segment unchanged, `unreliable` that what it returned
# is suspect -- see CLAUDE.md's operation conventions.
FLAG_KEYS = ("unreliable", "degenerate")

NO_IMAGE = "no image"
NO_MASK_INFO = "no recorded mask info"

# What to do about each kind of missing input, for log_no_input's one line per reason.
_NO_INPUT_HINTS = {NO_IMAGE: "run download_images first", NO_MASK_INFO: "re-segment to record it"}

# Seconds between a driver's progress lines (see Progress).
PROGRESS_INTERVAL = 30


class NoInput(Exception):
    """Nothing to work from yet: no image, or no upstream mask."""


def no_mask(part):
    """Return the `NoInput` reason for an occurrence missing `part`'s mask."""
    return f"no '{part}' mask"


def log_no_input(counts, where):
    """Log one warning per kind of missing input.

    Args:
        counts: `{reason: number of occurrences}`.
        where: What was running, for the message, e.g. `"run_segments part 'organism'"`.
    """
    for reason, count in sorted(Counter(counts).items()):
        hint = _NO_INPUT_HINTS.get(reason, "segment that part first")
        logger.warning("%s: %d occurrence(s) with %s yet -- %s", where, count, reason, hint)


class Tally:
    """What a driver counts while walking occurrences, and the reliability flags it saw."""

    def __init__(self, attempted=0):
        self.attempted = attempted
        self.processed = 0
        self.skipped = 0
        self.no_input = 0
        self.failed = 0
        self.failures = []
        self.flags = {}

    def record_failure(self, occurrence_id, error, **extra):
        """Count one failed occurrence and keep it.

        Args:
            occurrence_id: The item that failed.
            error: The exception or message.
            **extra: Anything else to trace it by, e.g. the `url` a download was given.
        """
        self.failed += 1
        self.failures.append({"occurrence_id": str(occurrence_id), "error": str(error), **extra})

    def record_flags(self, info):
        """Count the reliability flags (`FLAG_KEYS`) in an operation's info dict."""
        for key in FLAG_KEYS:
            if (info or {}).get(key):
                self.flags[key] = self.flags.get(key, 0) + 1

    def summary(self, **extra):
        """Return the dict a driver returns.

        Args:
            **extra: Driver-specific entries, e.g. `previously_failed` or `copied`.

        Returns:
            `attempted`, `processed`, `skipped`, `no_input`, `failed`, `failures`, `flags`, plus `extra`.
        """
        return {
            "attempted": self.attempted,
            "processed": self.processed,
            "skipped": self.skipped,
            "no_input": self.no_input,
            "failed": self.failed,
            "failures": self.failures,
            "flags": self.flags,
            **extra,
        }


def _duration(seconds):
    """Format seconds coarsely: `45s`, `12m 5s`, `2h 0m`."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


class Progress:
    """A time-throttled progress line with a rate and an ETA, for a per-item loop.

    The rate is taken over the most recent `WINDOW` items.

    Args:
        total: Items this loop will step through: the pending work, not everything considered.
        label: What is running, e.g. `"run_segments part 'organism'"`.
        tallies: `Tally`s whose `failed` counts are reported alongside.
        interval: Seconds between lines; None uses `PROGRESS_INTERVAL`, 0 logs only the closing line.
        log: Logging callable, so a line carries the caller's module name.
        clock: Monotonic clock, for tests.
    """

    WINDOW = 50

    def __init__(self, total, label, tallies=(), interval=None, log=None, clock=None):
        self.total = total
        self.label = label
        self.tallies = list(tallies)
        self.interval = PROGRESS_INTERVAL if interval is None else interval
        self.log = log or logger.info
        self.clock = clock or time.monotonic
        self.done = 0
        self.start = self.clock()
        self._last_log = self.start
        self._recent = deque([self.start], maxlen=self.WINDOW + 1)

    def step(self):
        """Count one finished item, and log a line if the interval has passed."""
        now = self.clock()
        self.done += 1
        self._recent.append(now)
        if self.interval and now - self._last_log >= self.interval and self.done < self.total:
            self._last_log = now
            self.log("%s", self.line())

    def seconds_per_item(self):
        """Return seconds per item over the recent window, or None before anything has finished."""
        if self.done == 0:
            return None
        return (self._recent[-1] - self._recent[0]) / (len(self._recent) - 1)

    def line(self):
        """Return the progress line as it stands."""
        pieces = [f"{self.label}: {self.done:,}/{self.total:,}"]
        if self.total:
            pieces[0] += f" ({100 * self.done / self.total:.1f}%)"
        rate = self.seconds_per_item()
        if rate is not None:
            pieces.append(f"{rate:.2g}s/item")
            if self.done < self.total:
                pieces.append(f"~{_duration(rate * (self.total - self.done))} left")
        failed = sum(tally.failed for tally in self.tallies)
        if failed:
            pieces.append(f"{failed:,} failed")
        return ", ".join(pieces)

    def finish(self):
        """Log a closing line if anything was stepped through, and return the elapsed seconds."""
        elapsed = self.clock() - self.start
        if self.total:
            self.log("%s, done in %s", self.line(), _duration(elapsed))
        return round(elapsed, 1)
