"""
What a project contains, read off what is actually there.

`summarize` returns data rather than printing it, which is what makes it
assertable -- and what makes it the natural end-to-end check that each stage of
a pipeline wrote what it claimed. It also stands behind the layout rule that
nothing creates anything: a project directory is an honest account of what has
been done to it, so a summary of a fresh project must report zeros without
conjuring the folders those zeros describe.
"""

import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.project.summarize import summarize
from helpers.models import ThresholdModel


def test_a_metadata_only_project_reports_no_images_or_masks(metadata_project):
    summary = summarize(metadata_project)

    assert summary["occurrences"] == 8
    assert summary["images"] == 0
    assert summary["parts"] == {}
    assert summary["runs"] == {}
    assert summary["metrics"] == {}


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
