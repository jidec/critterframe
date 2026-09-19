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

import json

import numpy as np
import pytest

from critterframe.project import paths
from critterframe.visualization import figures
from critterframe.visualization.pipeline import (
    NullReport,
    PanelFanout,
    Report,
    open_report,
    panel_sink,
    resolve_sample,
)

IDS = [f"specimen{index}" for index in range(30)]


def panel(value=200):
    return np.full((40, 60, 3), value, np.uint8)


def a_report(tmp_path, sample=("specimen0", "specimen1"), part="organism", **kwargs):
    return Report(tmp_path, "segments", "abc123", part=part, visualize=list(sample),
                  **kwargs).begin(IDS)


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
# Report: the main grid
# ---------------------------------------------------------------------------


def test_a_report_only_wants_its_sample(tmp_path):
    report = a_report(tmp_path)
    assert report.wants("specimen0") is True
    assert report.wants("specimen9") is False


def test_panels_are_collected_per_item_and_stage(tmp_path):
    report = a_report(tmp_path)
    report.collect("specimen0", "orientation", panel())
    report.collect("specimen0", "crop", panel())
    report.collect("specimen1", "orientation", panel())

    rows, labels = report.rows()
    assert labels == ["specimen0", "specimen1"]
    assert len(rows[0]) == 2


def test_a_panel_for_an_unwanted_item_is_ignored(tmp_path):
    """So code with no Segment can call report.panel() unconditionally."""
    report = a_report(tmp_path)
    report.panel("specimen9", "mask", panel())
    assert report.rows() == ([], [])


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


def test_one_stage_saves_as_a_grid_of_specimens(tmp_path):
    report = a_report(tmp_path)
    for occurrence_id in ("specimen0", "specimen1"):
        report.collect(occurrence_id, "mask", panel())

    written = report.save()
    assert written.exists()
    assert written.parent == paths.pipeline_dir(tmp_path)


def test_several_stages_save_as_a_row_per_specimen(tmp_path):
    report = a_report(tmp_path)
    for occurrence_id in ("specimen0", "specimen1"):
        report.collect(occurrence_id, "orientation", panel())
        report.collect(occurrence_id, "mask", panel())

    assert report.save().exists()


def test_a_report_that_collected_nothing_writes_no_file(tmp_path):
    """"Nothing here draws one" is a normal outcome, and an empty file would be worse than none."""
    report = a_report(tmp_path)
    report.close()
    assert not paths.pipeline_dir(tmp_path).exists()


def test_the_filename_names_the_run_and_the_recipe(tmp_path):
    report = a_report(tmp_path)
    report.collect("specimen0", "mask", panel())
    assert report.save().name == "segments_abc123.jpg"


def test_a_non_default_part_is_in_the_filename_too(tmp_path):
    report = a_report(tmp_path, part="wing")
    report.collect("specimen0", "mask", panel())
    assert report.save().name == "segments__wing_abc123.jpg"


def test_a_second_save_with_nothing_new_writes_nothing(tmp_path):
    """What makes periodic checkpointing of the main grid cheap."""
    report = a_report(tmp_path)
    report.collect("specimen0", "mask", panel())
    assert report.save() is not None
    assert report.save() is None


def test_collecting_again_after_a_save_makes_the_next_save_write(tmp_path):
    report = a_report(tmp_path)
    report.collect("specimen0", "mask", panel())
    report.save()

    report.collect("specimen1", "mask", panel())
    assert report.save() is not None


def test_eligible_narrows_the_sample_but_not_the_items(tmp_path):
    """
    A multi-part run walks one todo list per part, but samples only what its own
    part still needs.
    """
    report = Report(tmp_path, "parts", "abc", part="wing", visualize=True)
    report.begin(IDS, eligible=["specimen3", "specimen4"])
    assert report.wants("specimen3")
    assert not report.wants("specimen5")


# ---------------------------------------------------------------------------
# Report: checkpoints
# ---------------------------------------------------------------------------


def run_through(report, items, stage="mask"):
    for item in items:
        report.collect(item, stage, panel())
        report.done(item)


def test_visualize_every_writes_a_file_per_window(tmp_path):
    items = IDS[:6]
    report = Report(tmp_path, "segments", "abc", visualize=1, visualize_every=2).begin(items)
    run_through(report, items)
    report.close()

    names = sorted(path.name for path in paths.pipeline_dir(tmp_path).glob("*__at*.jpg"))
    assert names == ["segments_abc__at00000002.jpg", "segments_abc__at00000004.jpg",
                     "segments_abc__at00000006.jpg"]


def test_a_trailing_partial_window_is_written_on_close(tmp_path):
    items = IDS[:5]
    report = Report(tmp_path, "segments", "abc", visualize=1, visualize_every=2).begin(items)
    run_through(report, items)
    report.close()

    assert paths.pipeline_file_path(tmp_path, "segments", "abc", suffix="at00000005").exists()


def test_explicit_ids_follow_their_specimen_into_windows_without_warning(tmp_path, caplog):
    items = IDS[:6]
    with caplog.at_level("WARNING"):
        report = Report(tmp_path, "segments", "abc", visualize=["specimen3"],
                        visualize_every=2).begin(items)
        run_through(report, items)
        report.close()

    windows = list(paths.pipeline_dir(tmp_path).glob("*__at*.jpg"))
    assert [path.name for path in windows] == ["segments_abc__at00000004.jpg"]
    assert "isn't processing" not in caplog.text


def test_an_explicit_checkpoint_snapshots_then_resets(tmp_path):
    """The per-epoch shape: the same specimens, redrawn each round."""
    report = Report(tmp_path, "train", "abc", visualize=["specimen0"]).begin(["specimen0"])

    report.panel("specimen0", "prediction", panel())
    first = report.checkpoint("epoch0001")
    assert first.name == "train_abc__epoch0001.jpg"
    assert report.rows() == ([], [])

    report.panel("specimen0", "prediction", panel(10))
    assert report.checkpoint("epoch0002").exists()
    assert report.checkpoint("epoch0003") is None


# ---------------------------------------------------------------------------
# Report: rank mode
# ---------------------------------------------------------------------------


def ranked(tmp_path, rank, values, keep=2):
    items = list(values)
    report = Report(tmp_path, "validate", "abc", visualize=keep, rank=rank).begin(items)
    for item, value in values.items():
        report.panel(item, "compare", panel())
        report.done(item, rank_value=value)
    return report


def test_rank_lowest_keeps_the_worst_in_order(tmp_path):
    report = ranked(tmp_path, "lowest", {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.1})
    _rows, labels = report.rows()
    assert [label.split()[0] for label in labels] == ["d", "b"]


def test_rank_highest_keeps_the_largest_in_order(tmp_path):
    report = ranked(tmp_path, "highest", {"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.1})
    _rows, labels = report.rows()
    assert [label.split()[0] for label in labels] == ["a", "c"]


def test_rank_mode_holds_no_more_than_it_keeps(tmp_path):
    values = {f"item{index}": float(index) for index in range(200)}
    report = ranked(tmp_path, "lowest", values, keep=3)
    assert len(report._kept_cells) == 3
    assert not report._pending


def test_an_item_with_no_rank_value_is_never_kept(tmp_path):
    report = ranked(tmp_path, "lowest", {"a": None, "b": float("nan"), "c": 0.4})
    _rows, labels = report.rows()
    assert [label.split()[0] for label in labels] == ["c"]


def test_an_unknown_rank_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Report(tmp_path, "validate", "abc", rank="worst")


# ---------------------------------------------------------------------------
# Report: figures, failures, sidecar
# ---------------------------------------------------------------------------


def test_a_figure_is_written_beside_the_grid(tmp_path):
    report = Report(tmp_path, "train", "abc").begin([])
    written = report.figure("curves", figures.line_chart({"loss": [3, 2, 1]}))
    assert written.name == "train_abc__curves.png"
    assert written.stat().st_size > 0


def test_an_image_array_is_a_figure_too(tmp_path):
    report = Report(tmp_path, "train", "abc").begin([])
    assert report.figure("overview", panel()).exists()


def test_the_sidecar_records_identity_sample_counts_and_files(tmp_path):
    items = ["specimen0", "specimen1"]
    report = Report(tmp_path, "download", "abc", visualize=True,
                    identity={"url_col": "image_url"}).begin(items)
    report.panel("specimen0", "thumbnail", panel())
    report.done("specimen0")
    report.failure("specimen1", ValueError("404"))
    report.done("specimen1")
    report.close()

    record = json.loads(paths.pipeline_report_path(tmp_path, "download", "abc")
                        .read_text(encoding="utf-8"))
    assert record["identity"] == {"url_col": "image_url"}
    assert record["counts"] == {"items": 2, "done": 2, "failed": 1}
    assert record["failures"] == [{"item": "specimen1", "error": "404"}]
    assert record["files"] == ["download_abc.jpg"]


def test_failures_alone_still_leave_a_sidecar(tmp_path):
    """A download where nothing decoded has no image to show, only what went wrong."""
    report = Report(tmp_path, "download", "abc").begin(["a"])
    report.failure("a", "timed out")
    report.done("a")
    report.close()
    assert paths.pipeline_report_path(tmp_path, "download", "abc").exists()


def test_a_report_closes_as_a_context_manager(tmp_path):
    with Report(tmp_path, "segments", "abc", visualize=["a"]).begin(["a"]) as report:
        report.panel("a", "mask", panel())
    assert paths.pipeline_file_path(tmp_path, "segments", "abc").exists()


# ---------------------------------------------------------------------------
# open_report / NullReport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("visualize", [False, None])
def test_visualization_off_opens_a_null_report(tmp_path, visualize):
    report = open_report(tmp_path, "segments", "abc", visualize=visualize)
    assert isinstance(report, NullReport)
    assert not report
    assert open_report(tmp_path, "segments", "abc", visualize=4)


def test_a_null_report_does_nothing_at_all(tmp_path):
    report = open_report(tmp_path, "segments", "abc", visualize=False).begin(IDS)
    assert report.sink("specimen0") is None
    report.panel("specimen0", "mask", panel())
    report.done("specimen0", rank_value=1.0)
    report.failure("specimen0", "boom")
    assert report.checkpoint("epoch0001") is None
    assert report.figure("curves", panel()) is None
    report.close()
    assert not paths.pipeline_dir(tmp_path).exists()


def test_visualize_every_without_visualize_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        open_report(tmp_path, "segments", "abc", visualize=False, visualize_every=5)
    assert "has no effect" in caplog.text


# ---------------------------------------------------------------------------
# panel_sink / PanelFanout
# ---------------------------------------------------------------------------


def test_an_occurrence_outside_the_sample_gets_no_sink(tmp_path):
    """
    Which is what makes emit_panel a no-op for the great majority of
    occurrences without every operation re-checking a flag.
    """
    report = a_report(tmp_path)
    assert report.sink("specimen0") is report
    assert report.sink("specimen9") is None
    assert panel_sink(report, "specimen0") is not None
    assert panel_sink(None, "specimen0") is None


def test_a_fanout_gives_every_report_the_shared_steps(tmp_path):
    """
    A multi-output run's shared preprocessing happened once but belongs on
    every part's sheet -- otherwise each part's grid would start mid-recipe.
    """
    head = Report(tmp_path, "parts", "aaa", part="head", visualize=["specimen0"]).begin(IDS)
    wing = Report(tmp_path, "parts", "bbb", part="wing", visualize=["specimen0"]).begin(IDS)
    fanout = PanelFanout([head, wing])

    fanout.collect("specimen0", "remove_background", panel())
    assert head.rows()[0] and wing.rows()[0]


def test_a_fanout_of_nothing_is_falsey(tmp_path):
    assert not PanelFanout([])
    assert not PanelFanout([NullReport()])

