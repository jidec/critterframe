"""
A long run says how far it has got and how long is left, and a cached one stays quiet.

Progress is throttled on wall-clock time, so the clock ticks a full interval per
reading to make every item log; what is asserted is which work counts toward the total.
"""

import itertools
import logging
from types import SimpleNamespace

import pytest

import critterframe as cf
from critterframe import drivers
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8


def segment_run(project_path):
    return cf.run_segments(project_path, steps=[cf.segment(ThresholdModel())],
                           visualize=False)["organism"]


def tick_an_interval_per_reading(monkeypatch):
    ticks = itertools.count(step=drivers.PROGRESS_INTERVAL)
    monkeypatch.setattr(drivers, "time", SimpleNamespace(monotonic=lambda: next(ticks)))


def progress_lines(caplog):
    return [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("run_segments part(s) organism:")]


def test_a_run_logs_progress_over_its_pending_work(image_project, monkeypatch, caplog):
    tick_an_interval_per_reading(monkeypatch)
    with caplog.at_level(logging.INFO):
        result = segment_run(image_project)

    lines = progress_lines(caplog)
    assert len(lines) == SPECIMENS
    assert lines[0].startswith(f"run_segments part(s) organism: 1/{SPECIMENS}")
    assert f"{SPECIMENS}/{SPECIMENS} (100.0%)" in lines[-1]
    assert "done in" in lines[-1]
    assert result["elapsed_s"] >= 0


def test_a_fully_cached_rerun_logs_no_progress(segmented_project, monkeypatch, caplog):
    tick_an_interval_per_reading(monkeypatch)
    with caplog.at_level(logging.INFO):
        segment_run(segmented_project)
    assert progress_lines(caplog) == []
