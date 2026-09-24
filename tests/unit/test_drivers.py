"""
What every per-item driver shares: its progress line, throttled on wall-clock time,
with a rate over recent items and an ETA over pending work.
"""

from critterframe.drivers import Progress, Tally


class Clock:
    """A monotonic clock a test advances by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def progress(total, clock, interval=30, tallies=()):
    lines = []
    tracker = Progress(total, "run", tallies=tallies, interval=interval,
                       log=lambda fmt, *args: lines.append(fmt % args), clock=clock)
    return tracker, lines


def test_progress_logs_only_once_the_interval_has_passed():
    clock = Clock()
    tracker, lines = progress(100, clock)
    for _ in range(10):
        clock.now += 2
        tracker.step()
    assert lines == []

    clock.now += 11
    tracker.step()
    assert len(lines) == 1
    assert lines[0].startswith("run: 11/100 (11.0%)")


def test_progress_estimates_from_the_pending_total():
    clock = Clock()
    tracker, lines = progress(10, clock, interval=1)
    for _ in range(4):
        clock.now += 60
        tracker.step()
    assert tracker.seconds_per_item() == 60
    assert "~6m 0s left" in lines[-1]


def test_a_slow_first_item_ages_out_of_the_rate():
    clock = Clock()
    tracker, _lines = progress(1000, clock)
    clock.now += 600
    tracker.step()
    for _ in range(Progress.WINDOW):
        clock.now += 1
        tracker.step()
    assert tracker.seconds_per_item() == 1


def test_progress_reports_failures_from_its_tallies():
    clock = Clock()
    tally = Tally()
    tally.record_failure("a", "boom")
    tracker, lines = progress(2, clock, interval=1, tallies=[tally])
    clock.now += 5
    tracker.step()
    assert lines[-1].endswith("1 failed")


def test_finish_always_logs_and_returns_the_elapsed_time():
    clock = Clock()
    tracker, lines = progress(2, clock, interval=0)
    clock.now += 3
    tracker.step()
    clock.now += 3
    tracker.step()
    assert lines == []
    assert tracker.finish() == 6.0
    assert lines == ["run: 2/2 (100.0%), 3s/item, done in 6s"]


def test_nothing_to_do_logs_nothing():
    tracker, lines = progress(0, Clock())
    assert tracker.finish() == 0.0
    assert lines == []
