"""
Whose panels get laid out, and where the resulting sheet lands.

The contract this file protects: a pipeline visualization is a SAMPLE-level
summary of how processing behaved, and there is no per-occurrence file mode. A
10,000-occurrence run cannot be inspected as 10,000 files, so every form of
`visualize=` samples -- and the sample is deterministic, which is what lets two
versions of a recipe be compared cell by cell instead of specimen by specimen.

The other half is subtler: a grid can only show work that HAPPENED. A fully
cached rerun draws nothing, because there were no panels to collect -- which is
what `force=True` is for.
"""

import numpy as np
import pytest

from critterframe.project import paths
from critterframe.visualization.pipeline import (
    PanelFanout,
    RunReport,
    panel_sink,
    resolve_sample,
    run_report,
)

IDS = [f"specimen{index}" for index in range(30)]


def panel(value=200):
    return np.full((40, 60, 3), value, np.uint8)


def a_report(tmp_path, sample=("specimen0", "specimen1"), part="organism"):
    return RunReport(tmp_path, "segments", "abc123", part, list(sample))


# ---------------------------------------------------------------------------
# resolve_sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("visualize", [False, None])
def test_no_visualization_is_the_default_answer(visualize):
    assert resolve_sample(IDS, visualize) is None


def test_true_takes_a_default_sized_sample():
    sample = resolve_sample(IDS, True)
    assert 0 < len(sample) <= 25


def test_a_number_takes_that_many():
    assert len(resolve_sample(IDS, 6)) == 6


def test_the_sample_is_the_same_two_runs_running():
    """
    THE property. Two versions of a recipe must show the SAME specimens, or a
    changed method and a changed specimen look identical in the comparison.
    """
    assert resolve_sample(IDS, 6) == resolve_sample(IDS, 6)


def test_explicit_ids_follow_one_known_difficult_specimen():
    assert resolve_sample(IDS, ["specimen3", "specimen7"]) == ["specimen3",
                                                               "specimen7"]


def test_naming_an_occurrence_this_run_is_not_processing_is_ignored_loudly(caplog):
    """
    Not an error -- pointing `visualize=` at a list that spans several runs is
    reasonable -- but silence would leave you waiting for a panel that was
    never going to appear.
    """
    with caplog.at_level("WARNING"):
        sample = resolve_sample(IDS, ["specimen3", "ghost"])
    assert sample == ["specimen3"]
    assert "isn't processing" in caplog.text


def test_asking_for_more_than_the_run_covers_gives_what_there_is():
    assert len(resolve_sample(["a", "b"], 25)) == 2


def test_a_run_with_nothing_to_do_has_nothing_to_show():
    """
    A grid can only show work that happened. This is the empty case behind
    "a fully-cached rerun writes no grid".
    """
    assert resolve_sample([], 6) == []


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------


def test_a_report_with_no_sample_is_falsey(tmp_path):
    """So a run can skip building panels at all with one `if`."""
    assert not RunReport(tmp_path, "segments", "abc", "organism", [])
    assert a_report(tmp_path)


def test_a_report_only_wants_its_sample(tmp_path):
    report = a_report(tmp_path)
    assert report.wants("specimen0") is True
    assert report.wants("specimen9") is False


def test_panels_are_collected_per_occurrence_and_stage(tmp_path):
    report = a_report(tmp_path)
    report.collect("specimen0", "orientation", panel())
    report.collect("specimen0", "crop", panel())
    report.collect("specimen1", "orientation", panel())

    rows, labels = report.rows()
    assert labels == ["specimen0", "specimen1"]
    assert len(rows[0]) == 2


def test_a_stage_no_one_reached_leaves_a_hole_rather_than_shifting_the_row(tmp_path):
    """
    Which keeps every column under its own heading -- a shifted row would
    silently relabel every panel after the gap.
    """
    report = a_report(tmp_path)
    report.collect("specimen0", "orientation", panel())
    report.collect("specimen0", "crop", panel())
    report.collect("specimen1", "crop", panel())

    rows, _labels = report.rows()
    assert rows[1][0] is None


def test_an_occurrence_that_produced_nothing_is_left_out(tmp_path):
    report = a_report(tmp_path)
    report.collect("specimen0", "orientation", panel())

    _rows, labels = report.rows()
    assert labels == ["specimen0"]


def test_one_stage_saves_as_a_grid_of_specimens(tmp_path):
    """The view you scan for outliers."""
    report = a_report(tmp_path)
    for occurrence_id in ("specimen0", "specimen1"):
        report.collect(occurrence_id, "mask", panel())

    written = report.save()
    assert written.exists()
    assert written.parent == paths.pipeline_dir(tmp_path)


def test_several_stages_save_as_a_row_per_specimen(tmp_path):
    """The view that shows WHERE in a recipe something went wrong."""
    report = a_report(tmp_path)
    for occurrence_id in ("specimen0", "specimen1"):
        report.collect(occurrence_id, "orientation", panel())
        report.collect(occurrence_id, "mask", panel())

    assert report.save().exists()


def test_a_report_that_collected_nothing_writes_no_file(tmp_path, caplog):
    """
    "No operation in this recipe draws one" is a normal outcome, not a failure
    -- and an empty file would be worse than none.
    """
    with caplog.at_level("INFO"):
        assert a_report(tmp_path).save() is None
    assert not paths.pipeline_dir(tmp_path).exists()


def test_the_filename_names_the_run_and_the_recipe(tmp_path):
    """
    One file per run+recipe, so rerunning a changed recipe writes a second
    sheet rather than overwriting the evidence of the first.
    """
    report = a_report(tmp_path)
    report.collect("specimen0", "mask", panel())
    assert report.save().name == "segments_abc123.jpg"


def test_a_non_default_part_is_in_the_filename_too(tmp_path):
    """
    So a multi-part run gets one distinct sheet per part, while a plain
    whole-organism run keeps the obvious name.
    """
    report = a_report(tmp_path, part="wing")
    report.collect("specimen0", "mask", panel())
    assert report.save().name == "segments__wing_abc123.jpg"


# ---------------------------------------------------------------------------
# panel_sink / PanelFanout
# ---------------------------------------------------------------------------


def test_an_occurrence_outside_the_sample_gets_no_sink(tmp_path):
    """
    Which is what makes emit_panel a no-op for the great majority of
    occurrences without every operation re-checking a flag.
    """
    report = a_report(tmp_path)
    assert panel_sink(report, "specimen0") is not None
    assert panel_sink(report, "specimen9") is None
    assert panel_sink(None, "specimen0") is None


def test_a_fanout_gives_every_report_the_shared_steps(tmp_path):
    """
    A multi-output run's shared preprocessing happened once but belongs on
    every part's sheet -- otherwise each part's grid would start mid-recipe.
    """
    head = RunReport(tmp_path, "parts", "aaa", "head", ["specimen0"])
    wing = RunReport(tmp_path, "parts", "bbb", "wing", ["specimen0"])
    fanout = PanelFanout([head, wing])

    fanout.collect("specimen0", "remove_background", panel())
    assert head.rows()[0] and wing.rows()[0]


def test_a_fanout_of_nothing_is_falsey(tmp_path):
    assert not PanelFanout([])


def test_run_report_returns_none_when_visualization_is_off(tmp_path):
    assert run_report(tmp_path, "segments", "abc", "organism", IDS, False) is None
    assert run_report(tmp_path, "segments", "abc", "organism", IDS, 4) is not None
