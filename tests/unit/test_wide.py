"""
The wide view: long is what the metric log is, wide is a view built for whoever
is about to read it. Column naming, "newest wins" and filtering are part of that
view, and none of them touches what is stored.

The rule with teeth here: a NaN never passes a filter.
"""

import pandas as pd
import pytest

from critterframe.records.occurrences import ID_COL
from critterframe.wide import _to_millimetres, apply_filters, column_name, metric_units, metrics_wide
from helpers.stored import store_values


# ---------------------------------------------------------------------------
# column_name
# ---------------------------------------------------------------------------


def test_a_column_carries_run_part_and_metric():
    """
    All three vary independently and any two can collide -- the same metric on
    head and thorax, or under two differently configured runs.
    """
    assert column_name("traits", "organism", "body_length") == "traits__organism__body_length"


def test_a_dict_valued_metric_gets_one_column_per_key():
    assert column_name("traits", "organism", "centroid", "x") == "traits__organism__centroid__x"


def test_the_same_metric_on_two_parts_does_not_collide():
    assert column_name("traits", "head", "length") != column_name("traits", "thorax", "length")


# ---------------------------------------------------------------------------
# metrics_wide
# ---------------------------------------------------------------------------


def test_wide_is_one_row_per_occurrence(metadata_project):
    store_values(metadata_project, {"specimen0": 10.0, "specimen1": 20.0})
    wide = metrics_wide(metadata_project)

    assert wide.columns[0] == ID_COL
    assert len(wide) == 2
    assert wide.set_index(ID_COL)["traits__organism__body_length"]["specimen1"] == 20.0


def test_the_newest_value_wins(metadata_project):
    """
    A deliberate force=True rerun, or the same recipe rerun after a parameter
    change. Nothing is deleted -- the older rows stay in the long table with
    their own run ids, which is where to look when two numbers disagree.
    """
    store_values(metadata_project, {"specimen0": 10.0})
    store_values(metadata_project, {"specimen0": 99.0})

    wide = metrics_wide(metadata_project)
    assert len(wide) == 1
    assert wide["traits__organism__body_length"].iloc[0] == 99.0


def test_a_dict_value_becomes_several_columns(metadata_project):
    store_values(metadata_project, {"specimen0": {"x": 1.0, "y": 2.0}}, metric_name="centroid")
    wide = metrics_wide(metadata_project)
    assert wide["traits__organism__centroid__x"].iloc[0] == 1.0
    assert wide["traits__organism__centroid__y"].iloc[0] == 2.0


def test_two_runs_of_the_same_metric_stay_apart(metadata_project):
    """
    Two differently-configured measurements of one trait must stay
    distinguishable in the export rather than overwriting each other.
    """
    store_values(metadata_project, {"specimen0": 10.0}, run_name="traits")
    store_values(metadata_project, {"specimen0": 11.0}, run_name="traits_v2")

    wide = metrics_wide(metadata_project)
    assert wide["traits__organism__body_length"].iloc[0] == 10.0
    assert wide["traits_v2__organism__body_length"].iloc[0] == 11.0


def test_wide_can_be_narrowed_by_run_part_and_metric(metadata_project):
    store_values(metadata_project, {"specimen0": 10.0}, run_name="traits")
    store_values(metadata_project, {"specimen0": 0.5}, run_name="qc", metric_name="blur_variance")

    assert metrics_wide(metadata_project, run_names=["qc"]).columns.tolist() == [
        ID_COL,
        "qc__organism__blur_variance",
    ]
    assert metrics_wide(metadata_project, metric_names=["body_length"]).shape[1] == 2
    assert metrics_wide(metadata_project, parts=["wing"]).columns.tolist() == [ID_COL]


def test_an_unmeasured_project_gives_back_just_the_id_column(metadata_project):
    """
    So every caller's empty case is the same shape rather than a crash.
    """
    empty = metrics_wide(metadata_project)
    assert empty.empty
    assert empty.columns.tolist() == [ID_COL]


# ---------------------------------------------------------------------------
# metric_units
# ---------------------------------------------------------------------------


def test_units_are_reported_per_column(metadata_project):
    store_values(metadata_project, {"specimen0": 10.0}, unit="px")
    store_values(
        metadata_project, {"specimen0": 0.5}, run_name="qc", metric_name="mask_fraction", unit="fraction"
    )

    assert metric_units(metadata_project) == {
        "traits__organism__body_length": "px",
        "qc__organism__mask_fraction": "fraction",
    }


def test_a_dict_metric_s_keys_share_the_parent_unit(metadata_project):
    store_values(metadata_project, {"specimen0": {"x": 1.0, "y": 2.0}}, metric_name="centroid", unit="px")
    units = metric_units(metadata_project)
    assert units["traits__organism__centroid__x"] == "px"
    assert units["traits__organism__centroid__y"] == "px"


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


def filterable():
    return pd.DataFrame(
        {
            ID_COL: ["a", "b", "c", "d"],
            "length": [10.0, 50.0, 90.0, None],
            "flag": ["usable", "cut_off", "usable", "usable"],
        }
    )


@pytest.mark.parametrize(
    "condition, expected",
    [
        ((">", 40), ["b", "c"]),
        ((">=", 50), ["b", "c"]),
        (("<", 50), ["a"]),
        (("<=", 50), ["a", "b"]),
        (("==", 50), ["b"]),
        (("!=", 50), ["a", "c"]),
    ],
)
def test_each_comparison_selects_what_it_says(condition, expected):
    assert apply_filters(filterable(), {"length": condition})[ID_COL].tolist() == expected


def test_membership_filters_take_a_container():
    kept = apply_filters(filterable(), {"flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["a", "c", "d"]
    excluded = apply_filters(filterable(), {"flag": ("not in", ["usable"])})
    assert excluded[ID_COL].tolist() == ["b"]


def test_a_callable_expresses_what_the_shorthand_cannot():
    kept = apply_filters(filterable(), {"length": lambda series: series.between(20, 80)})
    assert kept[ID_COL].tolist() == ["b"]


def test_conditions_are_anded_together():
    kept = apply_filters(filterable(), {"length": (">", 20), "flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["c"]


@pytest.mark.parametrize("condition", [(">", 0), ("<", 1000), ("!=", 1), ("not in", ["x"])])
def test_a_missing_value_never_passes(condition):
    """
    "This metric wasn't measured" must not quietly count as passing a !=
    test -- an unmeasured occurrence is not a verified-good one.
    """
    assert "d" not in apply_filters(filterable(), {"length": condition})[ID_COL].tolist()


def test_filtering_on_a_column_that_is_not_there_raises():
    """A typo should be loud, not silently hand back an empty export."""
    with pytest.raises(KeyError, match="filter column"):
        apply_filters(filterable(), {"lenght": (">", 1)})


def test_an_unsupported_operator_raises():
    with pytest.raises(ValueError, match="unsupported filter op"):
        apply_filters(filterable(), {"length": ("~=", 1)})


# ---------------------------------------------------------------------------
# _to_millimetres
# ---------------------------------------------------------------------------


def wide_frame():
    return pd.DataFrame(
        {
            ID_COL: ["a", "b"],
            "traits__organism__body_length": [100.0, 200.0],
            "traits__organism__area_px": [10000.0, 40000.0],
            "traits__organism__mean_lightness": [0.5, 0.6],
        }
    )


UNITS = {
    "traits__organism__body_length": "px",
    "traits__organism__area_px": "px2",
    "traits__organism__mean_lightness": "fraction",
}


def test_lengths_divide_once_and_areas_twice():
    converted = _to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 20.0}))
    assert converted["traits__organism__body_length_mm"].tolist() == [10.0, 10.0]
    assert converted["traits__organism__area_px_mm2"].tolist() == [100.0, 100.0]


def test_the_converted_column_is_renamed_with_its_new_unit():
    """
    A CSV can't carry units in its header any other way, and two exports of one
    project differing only in units would otherwise be indistinguishable once
    the file is open in something else.
    """
    converted = _to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 10.0}))
    assert "traits__organism__body_length" not in converted.columns
    assert "traits__organism__body_length_mm" in converted.columns


def test_a_column_with_no_length_in_it_is_left_alone():
    """A fraction, a category, an embedding, a laplacian variance."""
    converted = _to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 10.0}))
    assert converted["traits__organism__mean_lightness"].tolist() == [0.5, 0.6]


def test_an_uncalibrated_occurrence_gets_nan_not_pixels():
    """
    The same number meaning something entirely different in the same column is
    the failure this prevents.
    """
    converted = _to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0}))
    assert converted["traits__organism__body_length_mm"].tolist()[0] == 10.0
    assert pd.isna(converted["traits__organism__body_length_mm"].tolist()[1])


def test_the_scale_rides_along_in_the_export():
    """
    A millimetre in the table is only as good as the calibration behind it, and
    someone reading the CSV a year later has to be able to see which one.
    """
    converted = _to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 20.0}))
    assert converted["px_per_mm"].tolist() == [10.0, 20.0]


def test_nothing_convertible_leaves_the_frame_untouched(caplog):
    frame = pd.DataFrame({ID_COL: ["a"], "traits__organism__mean_lightness": [0.5]})
    with caplog.at_level("WARNING"):
        converted = _to_millimetres(frame, UNITS, pd.Series({"a": 10.0}))
    assert converted.equals(frame)
    assert "nothing to" in caplog.text
