"""
What each operation reported about an occurrence is kept, so it can be filtered on.

A segmentation run stores every step's scalar info on the mask it helped make,
and `mask_info()` brings that into the metrics table. A metric run stores each
transform's info as a sibling `transform_info` row beside the values, which
export leaves out unless asked but can always filter on.
"""

import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from critterframe.records.metrics import TRANSFORM_INFO_UNIT, load_metrics
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8


def segment_parts(project_path, model=None):
    return cf.run_segments(
        project_path, run_name="body_parts", from_part="organism",
        shared_steps=[cf.crop_to_mask(), cf.orient()],
        outputs={"head": [cf.segment(model or ThresholdModel())],
                 "tail": [cf.segment(model or ThresholdModel())]},
        visualize=False)


# ---------------------------------------------------------------------------
# Segmentation: info on the mask row, brought over by mask_info()
# ---------------------------------------------------------------------------


def test_every_step_s_info_is_stored_on_each_part_s_mask(segmented_project):
    segment_parts(segmented_project)

    head = mask_records.mask_lookup(segmented_project, part="head")
    tail = mask_records.mask_lookup(segmented_project, part="tail")
    for occurrence_id in head:
        head_info = mask_records.mask_info(head[occurrence_id])
        tail_info = mask_records.mask_info(tail[occurrence_id])
        assert set(head_info) == {"crop_to_mask", "orient", "segment"}
        assert "unreliable" in head_info["orient"]
        assert head_info["orient"] == tail_info["orient"]   # one shared frame
        assert "area_fraction" in head_info["segment"]


def test_mask_info_makes_them_filterable_columns(segmented_project):
    segment_parts(segmented_project)
    result = cf.run_metrics(segmented_project, metrics=[cf.mask_info()],
                            part="head", visualize=False)["head"]
    assert result["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False,
                                 filters={"mask_info__head__mask_info__orient__unreliable":
                                          ("==", False)})
    assert "mask_info__head__mask_info__segment__area_fraction" in exported.columns
    assert "mask_info__head__mask_info__orient__unreliable" in exported.columns


def test_operations_narrows_what_mask_info_copies(segmented_project):
    segment_parts(segmented_project)
    cf.run_metrics(segmented_project, metrics=[cf.mask_info(operations=["orient"])],
                   part="head", visualize=False)
    columns = cf.export_metrics(segmented_project, path=False).columns
    assert any("__orient__" in column for column in columns)
    assert not any("__segment__" in column for column in columns)


def test_mask_info_goes_stale_when_the_part_is_resegmented(segmented_project):
    segment_parts(segmented_project)
    cf.run_metrics(segmented_project, metrics=[cf.mask_info()], part="head",
                   visualize=False)

    segment_parts(segmented_project, model=ThresholdModel(cutoff=90))
    columns = cf.export_metrics(segmented_project, path=False).columns
    assert not any(column.startswith("mask_info__head") for column in columns)


def test_a_mask_with_no_recorded_info_is_no_input(segmented_project):
    [row] = [mask_records.mask_lookup(segmented_project)[occurrence_id]
             for occurrence_id in sorted(mask_records.mask_lookup(segmented_project))[:1]]
    mask_records.save_masks(segmented_project, [mask_records.make_mask_row(
        row["occurrence_id"], mask_records.decode_mask(row),
        recipe_hash=row["recipe_hash"])])

    result = cf.run_metrics(segmented_project, metrics=[cf.mask_info()],
                            visualize=False)["organism"]
    assert (result["no_input"], result["failed"]) == (1, 0)


# ---------------------------------------------------------------------------
# Metric runs: transform info as sibling rows
# ---------------------------------------------------------------------------


def measure(project_path, run_name="body_dimensions", transforms=None, **kwargs):
    return cf.run_metrics(
        project_path, run_name=run_name,
        transforms=transforms or [cf.remove_appendages(), cf.orient()],
        metrics=[cf.body_length(), cf.max_width()], visualize=False,
        **kwargs)["organism"]


def test_each_transform_s_info_is_stored_beside_the_values(segmented_project):
    measure(segmented_project)
    info = load_metrics(segmented_project,
                        metric_names=["orient", "remove_appendages"])
    assert len(info) == 2 * SPECIMENS
    assert set(info["unit"]) == {TRANSFORM_INFO_UNIT}
    assert all("removed_fraction" in value for value in
               info.loc[info["metric_name"] == "remove_appendages", "value"])


def test_export_leaves_transform_info_out_unless_asked(segmented_project):
    measure(segmented_project)
    default = cf.export_metrics(segmented_project, path=False).columns
    asked = cf.export_metrics(segmented_project, path=False,
                              transform_info=True).columns

    assert not any("__orient__" in column for column in default)
    assert "body_dimensions__organism__orient__unreliable" in asked
    assert "body_dimensions__organism__body_length" in default


def test_a_filter_can_use_transform_info_without_exporting_it(segmented_project):
    measure(segmented_project)
    exported = cf.export_metrics(
        segmented_project, path=False,
        filters={"body_dimensions__organism__orient__unreliable": ("==", False)})
    assert "body_dimensions__organism__orient__unreliable" not in exported.columns
    assert len(exported) <= SPECIMENS


def test_naming_it_in_metric_names_includes_it(segmented_project):
    measure(segmented_project)
    exported = cf.export_metrics(segmented_project, path=False,
                                 metric_names=["body_length", "orient"])
    assert "body_dimensions__organism__orient__unreliable" in exported.columns


def test_a_default_export_s_identity_does_not_see_the_info_rows(segmented_project):
    """Hidden columns are not part of what the export says it contains."""
    measure(segmented_project)
    cf.export_metrics(segmented_project, path=False)
    cf.export_metrics(segmented_project, path=False, transform_info=True)
    default, asked = cf.load_exports(segmented_project).to_dict("records")

    assert not any("__orient__" in column for column in default["columns"])
    assert [run["n_values"] for run in default["runs"]] == [2 * SPECIMENS]
    assert "transform_info" not in default["selection"]

    assert [run["n_values"] for run in asked["runs"]] == [4 * SPECIMENS]
    assert asked["selection"]["transform_info"] is True
    assert default["export_hash"] != asked["export_hash"]


def test_a_repeated_transform_gets_its_own_label(segmented_project):
    measure(segmented_project, transforms=[cf.orient(), cf.orient()])
    names = set(load_metrics(segmented_project)["metric_name"])
    assert {"orient", "orient_2"} <= names


def test_a_transform_named_like_a_metric_is_refused(segmented_project):
    with pytest.raises(ValueError, match="share a name with a metric"):
        cf.run_metrics(segmented_project, run_name="clash",
                       transforms=[cf.orient()],
                       metrics=[cf.body_length(name="orient")], visualize=False)


def test_a_run_copied_under_a_new_name_brings_its_info_along(segmented_project):
    measure(segmented_project, run_name="first")
    second = measure(segmented_project, run_name="second")
    assert second["copied"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False, transform_info=True)
    assert "second__organism__orient__unreliable" in exported.columns


def test_comparing_two_runs_ignores_their_transform_info(segmented_project):
    """Booleans and strings can't be differenced; they describe how, not what."""
    measure(segmented_project, run_name="first")
    measure(segmented_project, run_name="second", force=True)
    compared = cf.compare_metrics(segmented_project, "first", "second",
                                  visualize=False)
    assert set(compared["metric"]) == {"body_length", "max_width"}


def test_a_selection_rule_can_name_transform_info(segmented_project):
    measure(segmented_project)
    sure = cf.occurrences_matching(segmented_project, "body_dimensions",
                                   {"orient__unreliable": [False]})
    unsure = cf.occurrences_matching(segmented_project, "body_dimensions",
                                     {"orient__unreliable": [True]})
    assert len(sure) + len(unsure) == SPECIMENS
