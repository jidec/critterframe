"""
A collage is a render, not a run: it reads stored values and cutouts, measures
nothing, records no run, and is named by a hash over what it draws.

The hash covers the placements -- where every specimen lands -- so a stored value
that reorders specimens, a grown selection, or a resegmented mask each makes a new
collage, while an unchanged one is recognized and not drawn twice.
"""

import json

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records.runs import load_runs
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

LIGHTNESS = "traits__organism__mean_lightness"


def rainbow(project_path, **kwargs):
    kwargs.setdefault("layout", cf.collage.grid())
    kwargs.setdefault("arrange", [cf.collage.sort_by(LIGHTNESS)])
    return cf.render_collage(project_path, "rainbow", max_side=600, **kwargs)


def sidecar(result):
    path = result["path"].with_name(result["path"].name.replace(".png", paths.COLLAGE_SIDECAR_SUFFIX))
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_collage_writes_an_image_its_record_and_its_placements(measured_project):
    result = rainbow(measured_project)
    assert result["path"].exists()
    assert result["placed"] == result["processed"] == 8

    record = sidecar(result)
    placements = pd.read_csv(
        str(result["path"]).replace(".png", ".placements.csv"), dtype={"occurrence_id": str}
    )
    assert record["occurrences"]["count"] == 8
    assert record["counts"]["placed"] == len(placements) == 8
    assert record["columns"][LIGHTNESS]["run_name"] == "traits"
    assert [run["name"] for run in record["runs"]] == ["traits"]


def test_a_collage_records_no_run(measured_project):
    before = len(load_runs(measured_project))
    rainbow(measured_project)
    assert len(load_runs(measured_project)) == before


def test_an_identical_rerun_is_skipped(measured_project):
    first = rainbow(measured_project)
    second = rainbow(measured_project)
    assert second["path"] == first["path"]
    assert second["processed"] == 0 and second["skipped"] == 8


def test_force_draws_it_again(measured_project):
    rainbow(measured_project)
    assert rainbow(measured_project, force=True)["processed"] == 8


def test_a_value_that_reorders_specimens_moves_the_hash(measured_project):
    by_lightness = rainbow(measured_project)
    reversed_ = rainbow(measured_project, arrange=[cf.collage.sort_by(LIGHTNESS, descending=True)])
    by_length = rainbow(measured_project, arrange=[cf.collage.sort_by("traits__organism__body_length")])
    assert by_lightness["collage_hash"] != reversed_["collage_hash"]
    # Different specs that happen to place everyone the same way are one collage.
    assert len({by_lightness["collage_hash"], reversed_["collage_hash"], by_length["collage_hash"]}) >= 2


def test_a_smaller_selection_moves_the_hash(measured_project):
    every = rainbow(measured_project)
    some = rainbow(measured_project, occurrence_ids=["specimen0", "specimen1", "specimen2"])
    assert some["placed"] == 3
    assert some["collage_hash"] != every["collage_hash"]


def test_a_filter_narrows_the_collage_as_it_narrows_an_export(measured_project):
    filters = {LIGHTNESS: (">", -1.0), "species": ("==", "Anax junius")}
    result = rainbow(measured_project, filters=filters)
    exported = cf.export_metrics(measured_project, filters=filters, path=False)
    assert result["placed"] == len(exported)
    assert sidecar(result)["selection"]["filters"]["species"] == ["==", "Anax junius"]


def test_a_missing_value_leaves_the_item_out(measured_project):
    cf.run_metrics(
        measured_project,
        run_name="some",
        metrics=[cf.mask_area()],
        limit=2,
        visualize=False,
    )
    result = rainbow(measured_project, arrange=[cf.collage.sort_by("some__organism__mask_area")])
    assert result["placed"] == 2
    assert result["missing_value"] == 6


def test_an_unknown_column_says_what_it_looked_for(measured_project):
    with pytest.raises(KeyError, match="traits__organism__mean_lightnes"):
        rainbow(measured_project, arrange=[cf.collage.sort_by("traits__organism__mean_lightnes")])


def test_occurrence_columns_arrange_without_any_metric(segmented_project):
    result = cf.render_collage(
        segmented_project,
        "rings",
        cf.collage.rings(by_group=True),
        arrange=[cf.collage.group_by("species")],
        max_side=600,
    )
    assert result["placed"] == 8
    assert sidecar(result)["columns"] == {}


def test_a_resegmented_mask_renders_a_new_cutout_and_moves_the_hash(segmented_project):
    first = cf.render_collage(segmented_project, "plain", cf.collage.grid(), max_side=600)
    cf.run_segments(segmented_project, steps=[cf.segment(ThresholdModel(erode=3))], visualize=False)
    second = cf.render_collage(segmented_project, "plain", cf.collage.grid(), max_side=600)
    assert second["collage_hash"] != first["collage_hash"]
    assert sidecar(second)["cutouts"]["derivations"] != sidecar(first)["cutouts"]["derivations"]


def test_items_without_a_mask_are_counted_not_failed(image_project):
    result = cf.render_collage(image_project, "nothing", cf.collage.grid(), max_side=600)
    assert result["no_input"] == 8
    assert result["placed"] == 0
    assert result["failed"] == 0
    assert result["path"] is None


def test_a_jpeg_collage_needs_a_background(measured_project):
    with pytest.raises(ValueError, match="background"):
        rainbow(measured_project, extension="jpg")
    assert rainbow(measured_project, extension="jpg", background=(255, 255, 255))["path"].suffix == ".jpg"


def test_several_parts_are_one_item_each(measured_project):
    cf.run_segments(
        measured_project,
        part="half",
        from_part="organism",
        steps=[cf.segment(ThresholdModel(erode=4))],
        visualize=False,
    )
    result = cf.render_collage(
        measured_project,
        "parts",
        cf.collage.nested(cf.collage.grid(), cf.collage.grid()),
        arrange=[cf.collage.group_by("occurrence_id")],
        parts=["organism", "half"],
        max_side=600,
    )
    assert result["attempted"] == 16
    assert result["placed"] == 16
