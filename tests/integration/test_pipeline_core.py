"""
The five calls the package's own docstring opens with, run end to end.

    ingest -> segment -> measure -> export

Everything in `tests/unit/` checks one piece against its own contract. This
checks that the pieces still fit: that a mask written by a segmentation run is
the mask a metric run reads, that the metric names a recipe used are the column
names an export produces, and that the numbers arriving at the far end describe
the specimens that went in.

Harvested from `scripts/simple_tests/pipeline_synthetic_test.py`, which walks
the same path and prints what it finds.
"""

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records import masks as mask_records
from critterframe.records.runs import load_runs
from helpers.models import ThresholdModel
from helpers.synthetic import specimen_metadata, write_specimens

pytestmark = pytest.mark.slow

SPECIMENS = 8
METRICS = [cf.body_length(), cf.max_width(),
           cf.mask_area(name="area_px", unit="px2"), cf.mean_lightness()]


def test_a_project_can_be_built_from_nothing_but_a_folder_of_images(tmp_path):
    """
    Ingest, segment, measure, export -- and at each step the project directory
    grows only what that step needed. What a project contains is an honest
    account of what has been done to it.
    """
    images = tmp_path / "images"
    project = tmp_path / "project"
    ids = write_specimens(images, count=SPECIMENS)

    ingested = cf.ingest_images(project, images,
                                metadata=specimen_metadata(ids))
    assert ingested["saved"] == SPECIMENS
    assert paths.occurrences_path(project).exists()
    assert not paths.masks_path(project).exists()

    segmented = cf.run_segments(project, steps=[cf.segment(ThresholdModel())],
                                visualize=False)["organism"]
    assert segmented["processed"] == SPECIMENS
    assert paths.masks_path(project).exists()
    assert paths.runs_and_metrics_path(project).exists()   # the run record

    measured = cf.run_metrics(project, run_name="traits", metrics=METRICS,
                              visualize=False)["organism"]
    assert measured["processed"] == SPECIMENS

    exported = cf.export_metrics(project)
    assert len(exported) == SPECIMENS


def test_the_export_carries_one_column_per_run_part_and_metric(measured_project):
    exported = cf.export_metrics(measured_project)

    assert exported.columns[0] == "occurrence_id"
    assert "traits__organism__body_length" in exported.columns
    assert "traits__organism__area_px" in exported.columns
    assert len(exported) == SPECIMENS


def test_the_numbers_describe_the_specimens_that_went_in(measured_project):
    """
    Not a tautology: the drawn specimens are ellipses of a known size, and a
    body length of 3 or 3000 would mean the mask, the transform chain, or the
    coordinate frame had gone wrong somewhere between the store and the export.

    Area is asserted tightly and length loosely, and the reason is worth
    knowing: `orient()` picks the body axis by ASYMMETRY, and a drawn ellipse is
    very nearly symmetric, so on some of these the choice is genuinely noise and
    the specimen comes out measured across rather than along. That is correct
    behaviour on a shape with no real answer -- `compute_orientation` reports it
    as unreliable -- and it is why the synthetic fixtures are not the place to
    test orientation. Area does not care which way round the specimen sits.
    """
    exported = cf.export_metrics(measured_project)
    lengths = exported["traits__organism__body_length"]
    areas = exported["traits__organism__area_px"]

    assert areas.between(2000, 8000).all()
    assert exported["traits__organism__mean_lightness"].between(0, 1).all()

    # Either the long axis or the short one, never something in between and
    # never something absurd.
    assert lengths.between(30, 160).all()
    assert (exported["traits__organism__max_width"] > 0).all()


def test_occurrence_metadata_can_ride_along(measured_project):
    exported = cf.export_metrics(measured_project,
                                 occurrence_columns=["device", "species"])
    assert set(exported["device"]) == {"boxA", "boxB"}


def test_the_export_writes_a_csv_anything_can_read(measured_project, tmp_path):
    destination = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, destination)

    from_disk = pd.read_csv(destination)
    assert len(from_disk) == SPECIMENS
    assert from_disk["occurrence_id"].dtype == object


def test_units_come_out_beside_the_table(measured_project):
    """
    A CSV can't carry units in its header without mangling the column names, so
    they are available separately -- and every number in an export is in pixels,
    a fraction, or a category, which is not guessable from the name alone.
    """
    units = cf.export_units(measured_project)
    assert units["traits__organism__body_length"] == "px"
    assert units["traits__organism__area_px"] == "px2"


def test_every_step_leaves_a_record_of_itself(measured_project):
    """
    Two runs, both complete, with the recipe each executed stored in full --
    so the project stays readable years later even if the operation that
    produced it has since been renamed.
    """
    runs = load_runs(measured_project)

    assert len(runs) == 2
    assert set(runs["kind"]) == {"segment", "metric"}
    assert (runs["status"] == "complete").all()
    assert all(isinstance(recipe, dict) for recipe in runs["recipe"])
    assert runs["n_processed"].sum() == SPECIMENS * 2


def test_one_mask_per_occurrence_part(measured_project):
    stored = mask_records.load_masks(measured_project)
    assert len(stored) == SPECIMENS
    assert stored["part"].unique().tolist() == ["organism"]
    assert not stored.duplicated(subset=["occurrence_id", "part"]).any()


def test_masks_are_stored_in_the_original_image_s_coordinates(measured_project):
    """
    The invariant that makes parts and whole organisms comparable. Every stored
    mask is the size of the photograph it came from, whatever the recipe did to
    the working frame on the way.
    """
    from critterframe.storage.imagestore import ImageStore

    stored = mask_records.load_masks(measured_project)
    with ImageStore(measured_project, readonly=True) as images:
        for row in stored.itertuples(index=False):
            image = images.get(row.occurrence_id)
            assert (row.rle_height, row.rle_width) == image.shape[:2]


def test_a_transform_chain_changes_the_numbers_it_should(segmented_project):
    """
    Measuring an oriented, appendage-free segment is different work AND a
    different answer from measuring the raw mask -- which is what makes the
    chain part of the recipe rather than a preference.
    """
    cf.run_metrics(segmented_project, run_name="raw", metrics=[cf.body_length()],
                   visualize=False)
    cf.run_metrics(segmented_project, run_name="cleaned",
                   transforms=[cf.remove_appendages(), cf.orient()],
                   metrics=[cf.body_length()], visualize=False)

    exported = cf.export_metrics(segmented_project)
    assert not exported["raw__organism__body_length"].equals(
        exported["cleaned__organism__body_length"])


def test_a_project_survives_a_re_ingest_of_its_occurrences(measured_project,
                                                           tmp_path):
    """
    Masks and metrics are keyed by occurrence id, so occurrences still present
    keep everything derived from them. This is what makes a scheduled re-ingest
    safe on a source that grows.
    """
    source = tmp_path / "again.csv"
    pd.DataFrame({"occurrence_id": [f"specimen{index}" for index in range(8)],
                  "device": ["boxA"] * 8}).to_csv(source, index=False)

    cf.ingest_occurrences(measured_project, source)

    assert len(mask_records.load_masks(measured_project)) == SPECIMENS
    assert len(cf.export_metrics(measured_project)) == SPECIMENS
