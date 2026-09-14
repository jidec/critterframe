"""
What a project contains, read off what is actually there.

`summarize` returns data rather than printing it, which is what makes it
assertable -- and what makes it the natural end-to-end check that each stage of
a pipeline wrote what it claimed. It also stands behind the layout rule that
nothing creates anything: a project directory is an honest account of what has
been done to it, so a summary of a fresh project must report zeros without
conjuring the folders those zeros describe.
"""

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.project.summarize import summarize
from critterframe.records.occurrences import ID_COL, save_occurrences
from helpers.models import ThresholdModel


def test_a_metadata_only_project_reports_no_images_or_masks(metadata_project):
    summary = summarize(metadata_project)

    assert summary["occurrences"] == 8
    assert summary["images"] == 0
    assert summary["parts"] == {}
    assert summary["runs"] == {}
    assert summary["metrics"] == {}


def test_occurrence_preview_is_a_head_style_table(metadata_project):
    """
    Row index, header, aligned values, one row per previewed occurrence --
    the R head() look. head defaults to 6 like R's does; metadata_project has
    8 rows, so the preview covers 6 of them, not all 8.
    """
    preview = summarize(metadata_project)["occurrence_preview"]
    lines = preview.splitlines()

    assert ID_COL in lines[0]           # header row names the columns
    assert len(lines) == 1 + 6          # header + 6 previewed rows


def test_occurrence_preview_respects_a_custom_head_count(metadata_project):
    preview = summarize(metadata_project, head=2)["occurrence_preview"]
    assert len(preview.splitlines()) == 1 + 2


def test_occurrence_preview_is_empty_with_no_occurrences(tmp_path):
    """A project can exist (require_project only needs the table present)
    with zero rows -- nothing to preview, not an empty-looking table."""
    save_occurrences(tmp_path, pd.DataFrame({ID_COL: []}))
    assert summarize(tmp_path)["occurrence_preview"] == ""


def test_print_summary_prints_the_occurrence_preview(metadata_project, capsys):
    cf.print_summary(metadata_project)
    output = capsys.readouterr().out

    assert "occurrences (head of 6 of 8)" in output
    assert ID_COL in output


def test_summarizing_creates_nothing(metadata_project):
    """
    Reading a project must not leave folders behind for work nobody did --
    which is exactly what an unguarded image-store open would do.
    """
    summarize(metadata_project)
    assert not paths.masks_path(metadata_project).exists()
    assert not paths.visualizations_dir(metadata_project).exists()


def test_images_are_counted_once_ingested(image_project):
    assert summarize(image_project)["images"] == 8


def test_masks_are_counted_per_part(segmented_project):
    assert summarize(segmented_project)["parts"] == {"organism": 8}


def test_reference_masks_are_counted_separately(segmented_project):
    """
    Validation is comparison between two tables, so a summary that pooled them
    would hide whether there was anything to compare against.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(cutoff=120))],
                    reference=True, limit=3, visualize=False)

    summary = summarize(segmented_project)
    assert summary["parts"] == {"organism": 8}
    assert summary["reference_parts"] == {"organism": 3}


def test_runs_and_values_are_reported(measured_project):
    summary = summarize(measured_project)

    assert summary["runs"]["total"] == 2
    assert summary["runs"]["by_kind"] == {"metric": 1, "segment": 1}
    assert summary["runs"]["unfinished"] == 0
    assert summary["runs"]["latest"] == "traits"

    assert summary["metrics"]["values"] == 8 * 7
    assert summary["metrics"]["occurrences_measured"] == 8
    assert "body_length" in summary["metrics"]["names"]


def test_runs_are_broken_down_by_name(measured_project):
    """
    measured_project carries one segment run (default name 'segments') and one
    metric run ('traits'), neither ever rerun -- so each gets exactly one
    by_name row, and its current recipe is its only recipe.
    """
    by_name = {(row["kind"], row["name"]): row
               for row in summarize(measured_project)["runs"]["by_name"]}

    assert set(by_name) == {("segment", "segments"), ("metric", "traits")}
    for row in by_name.values():
        assert row["part"] == "organism"
        assert row["n_runs"] == 1
        assert row["current_recipe_hash"] == row["latest_recipe_hash"]
        assert row["description"].startswith(f"{row['kind']}:{row['name']}")


def test_a_rerun_metric_name_reports_its_history_and_current_pointer(measured_project):
    """
    Rerunning 'traits' under force=True moves the current pointer but keeps
    both runs in history -- by_name has to report both without losing either.
    """
    cf.run_metrics(measured_project, run_name="traits",
                   transforms=[cf.remove_appendages(), cf.orient()],
                   metrics=[cf.body_length()], force=True, visualize=False)

    by_name = {(row["kind"], row["name"]): row
               for row in summarize(measured_project)["runs"]["by_name"]}
    traits = by_name[("metric", "traits")]

    assert traits["n_runs"] == 2
    assert traits["current_recipe_hash"] == traits["latest_recipe_hash"]


def test_print_summary_lists_each_run_by_name(measured_project, capsys):
    cf.print_summary(measured_project)
    output = capsys.readouterr().out

    assert "'segments'" in output
    assert "'traits'" in output


def test_describe_run_prints_the_recipe_and_returns_the_row(measured_project, capsys):
    row = cf.describe_run(measured_project, name="traits")
    output = capsys.readouterr().out

    assert row["name"] == "traits"
    assert "body_length" in output
    assert "remove_appendages" in output


def test_describe_run_resolves_an_exact_run_id(measured_project):
    run_id = cf.load_runs(measured_project, name="traits").iloc[0]["run_id"]
    row = cf.describe_run(measured_project, run_id=int(run_id))
    assert row["run_id"] == run_id


def test_describe_run_raises_for_an_unknown_name(measured_project):
    with pytest.raises(KeyError):
        cf.describe_run(measured_project, name="does-not-exist")


def test_the_project_path_is_reported_as_a_string(measured_project):
    """So the summary is printable and serializable without a Path in it."""
    assert summarize(measured_project)["project_path"] == str(measured_project)


def test_summarizing_a_directory_that_is_not_a_project_raises(empty_project):
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        summarize(empty_project)


def test_print_summary_returns_what_it_printed(measured_project, capsys):
    printed = cf.print_summary(measured_project)
    output = capsys.readouterr().out

    assert printed == summarize(measured_project)
    assert "occurrences" in output
    assert "traits" in output
