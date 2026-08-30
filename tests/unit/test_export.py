"""
Getting data out: the wide view, and the two decisions it makes on the way.

The reshape is an OUTPUT decision, not a storage one -- long is what the metric
log is, wide is a view built for whoever is about to read it. Column naming,
"newest wins", dropping stale values, converting units, and filtering are all
part of that same view, which is why they live together and why none of them
touches what is stored.

The rules with teeth: filtering happens at export only and deletes nothing; a
NaN never passes a filter; and an occurrence with no calibration gets NaN
millimetres rather than an unconverted pixel value sitting in a column labelled
mm.
"""

import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.export import (
    apply_filters,
    column_name,
    metric_units,
    metrics_wide,
    occurrences_matching,
    to_millimetres,
)
from critterframe.metrics.annotation import annotate_flags
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.occurrences import ID_COL
from critterframe.records.runs import start_run
from critterframe.recipes import Recipe
from critterframe.metrics.dimensions import body_length


def store_values(project_path, values, run_name="traits", part="organism",
                 metric_name="body_length", unit="px"):
    """Append {occurrence_id: value} under a fresh run of `run_name`."""
    recipe = Recipe("metric", run_name, [body_length()], part=part)
    run_id = start_run(project_path, recipe)
    append_metrics(project_path, run_id, recipe.hash,
                   [make_metric_row(occurrence_id, part, metric_name, value,
                                    unit=unit)
                    for occurrence_id, value in values.items()])


# ---------------------------------------------------------------------------
# column_name
# ---------------------------------------------------------------------------


def test_a_column_carries_run_part_and_metric():
    """
    All three vary independently and any two can collide -- the same metric on
    head and thorax, or under two differently configured runs.
    """
    assert column_name("traits", "organism", "body_length") == \
        "traits__organism__body_length"


def test_a_dict_valued_metric_gets_one_column_per_key():
    assert column_name("traits", "organism", "centroid", "x") == \
        "traits__organism__centroid__x"


def test_the_same_metric_on_two_parts_does_not_collide():
    assert (column_name("traits", "head", "length")
            != column_name("traits", "thorax", "length"))


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
    store_values(metadata_project, {"specimen0": {"x": 1.0, "y": 2.0}},
                 metric_name="centroid")
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
    store_values(metadata_project, {"specimen0": 0.5}, run_name="qc",
                 metric_name="blur_variance")

    assert metrics_wide(metadata_project, run_names=["qc"]).columns.tolist() == [
        ID_COL, "qc__organism__blur_variance"]
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
    store_values(metadata_project, {"specimen0": 0.5}, run_name="qc",
                 metric_name="mask_fraction", unit="fraction")

    assert metric_units(metadata_project) == {
        "traits__organism__body_length": "px",
        "qc__organism__mask_fraction": "fraction",
    }


def test_a_dict_metric_s_keys_share_the_parent_unit(metadata_project):
    store_values(metadata_project, {"specimen0": {"x": 1.0, "y": 2.0}},
                 metric_name="centroid", unit="px")
    units = metric_units(metadata_project)
    assert units["traits__organism__centroid__x"] == "px"
    assert units["traits__organism__centroid__y"] == "px"


# ---------------------------------------------------------------------------
# to_millimetres
# ---------------------------------------------------------------------------


def wide_frame():
    return pd.DataFrame({
        ID_COL: ["a", "b"],
        "traits__organism__body_length": [100.0, 200.0],
        "traits__organism__area_px": [10000.0, 40000.0],
        "traits__organism__mean_lightness": [0.5, 0.6],
    })


UNITS = {
    "traits__organism__body_length": "px",
    "traits__organism__area_px": "px2",
    "traits__organism__mean_lightness": "fraction",
}


def test_lengths_divide_once_and_areas_twice():
    converted = to_millimetres(wide_frame(), UNITS,
                               pd.Series({"a": 10.0, "b": 20.0}))
    assert converted["traits__organism__body_length_mm"].tolist() == [10.0, 10.0]
    assert converted["traits__organism__area_px_mm2"].tolist() == [100.0, 100.0]


def test_the_converted_column_is_renamed_with_its_new_unit():
    """
    A CSV can't carry units in its header any other way, and two exports of one
    project differing only in units would otherwise be indistinguishable once
    the file is open in something else.
    """
    converted = to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 10.0}))
    assert "traits__organism__body_length" not in converted.columns
    assert "traits__organism__body_length_mm" in converted.columns


def test_a_column_with_no_length_in_it_is_left_alone():
    """A fraction, a category, an embedding, a laplacian variance."""
    converted = to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 10.0}))
    assert converted["traits__organism__mean_lightness"].tolist() == [0.5, 0.6]


def test_an_uncalibrated_occurrence_gets_nan_not_pixels():
    """
    The same number meaning something entirely different in the same column is
    the failure this prevents.
    """
    converted = to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0}))
    assert converted["traits__organism__body_length_mm"].tolist()[0] == 10.0
    assert pd.isna(converted["traits__organism__body_length_mm"].tolist()[1])


def test_the_scale_rides_along_in_the_export():
    """
    A millimetre in the table is only as good as the calibration behind it, and
    someone reading the CSV a year later has to be able to see which one.
    """
    converted = to_millimetres(wide_frame(), UNITS, pd.Series({"a": 10.0, "b": 20.0}))
    assert converted["px_per_mm"].tolist() == [10.0, 20.0]


def test_nothing_convertible_leaves_the_frame_untouched(caplog):
    frame = pd.DataFrame({ID_COL: ["a"], "traits__organism__mean_lightness": [0.5]})
    with caplog.at_level("WARNING"):
        converted = to_millimetres(frame, UNITS, pd.Series({"a": 10.0}))
    assert converted.equals(frame)
    assert "nothing to" in caplog.text


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


def filterable():
    return pd.DataFrame({
        ID_COL: ["a", "b", "c", "d"],
        "length": [10.0, 50.0, 90.0, None],
        "flag": ["usable", "cut_off", "usable", "usable"],
    })


@pytest.mark.parametrize("condition, expected", [
    ((">", 40), ["b", "c"]),
    ((">=", 50), ["b", "c"]),
    (("<", 50), ["a"]),
    (("<=", 50), ["a", "b"]),
    (("==", 50), ["b"]),
    (("!=", 50), ["a", "c"]),
])
def test_each_comparison_selects_what_it_says(condition, expected):
    assert apply_filters(filterable(), {"length": condition})[ID_COL].tolist() == expected


def test_membership_filters_take_a_container():
    kept = apply_filters(filterable(), {"flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["a", "c", "d"]
    excluded = apply_filters(filterable(), {"flag": ("not in", ["usable"])})
    assert excluded[ID_COL].tolist() == ["b"]


def test_a_callable_expresses_what_the_shorthand_cannot():
    kept = apply_filters(filterable(),
                         {"length": lambda series: series.between(20, 80)})
    assert kept[ID_COL].tolist() == ["b"]


def test_conditions_are_anded_together():
    kept = apply_filters(filterable(),
                         {"length": (">", 20), "flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["c"]


@pytest.mark.parametrize("condition", [(">", 0), ("<", 1000), ("!=", 1),
                                       ("not in", ["x"])])
def test_a_missing_value_never_passes(condition):
    """
    "This metric wasn't measured" must not quietly count as passing a !=
    test -- an unmeasured occurrence is not a verified-good one.
    """
    assert "d" not in apply_filters(filterable(),
                                    {"length": condition})[ID_COL].tolist()


def test_filtering_on_a_column_that_is_not_there_raises():
    """A typo should be loud, not silently hand back an empty export."""
    with pytest.raises(KeyError, match="filter column"):
        apply_filters(filterable(), {"lenght": (">", 1)})


def test_an_unsupported_operator_raises():
    with pytest.raises(ValueError, match="unsupported filter op"):
        apply_filters(filterable(), {"length": ("~=", 1)})


# ---------------------------------------------------------------------------
# export_metrics
# ---------------------------------------------------------------------------


def test_export_writes_a_csv_only_when_asked(measured_project, tmp_path):
    assert not list(tmp_path.glob("*.csv"))
    cf.export_metrics(measured_project)
    assert not list(tmp_path.glob("*.csv"))

    destination = tmp_path / "traits.csv"
    exported = cf.export_metrics(measured_project, destination)
    assert len(pd.read_csv(destination)) == len(exported)


def test_identifying_columns_come_first(measured_project):
    """Where anyone opening the CSV expects them."""
    exported = cf.export_metrics(measured_project,
                                 occurrence_columns=["device", "species"])
    assert exported.columns[:3].tolist() == [ID_COL, "device", "species"]


def test_a_subset_narrows_the_export(measured_project):
    cf.define_subset(measured_project, "boxA", column="device", values=["boxA"])
    assert len(cf.export_metrics(measured_project, subset="boxA")) == 4


def test_filters_narrow_the_export_and_delete_nothing(measured_project):
    """
    Filtering happens at export only. A judgement about degree stays revisable,
    which means the project still holds every row after a filtered export.
    """
    column = "traits__organism__body_length"
    everything = cf.export_metrics(measured_project)
    filtered = cf.export_metrics(measured_project,
                                 filters={column: (">", everything[column].median())})

    assert 0 < len(filtered) < len(everything)
    assert len(cf.export_metrics(measured_project)) == len(everything)


def test_export_units_reports_what_each_column_holds(measured_project):
    units = cf.export_units(measured_project)
    assert units["traits__organism__body_length"] == "px"
    assert units["traits__organism__area_px"] == "px2"
    assert units["traits__organism__mean_lightness"] == "fraction"


def test_exporting_from_a_directory_that_is_not_a_project_raises(empty_project):
    with pytest.raises(FileNotFoundError, match="isn't a CritterFrame project"):
        cf.export_metrics(empty_project)


def test_millimetres_need_a_calibration(measured_project):
    """
    Rather than handing back a table of empty millimetre columns, which reads
    as "these specimens are all unmeasurable".
    """
    with pytest.raises(ValueError, match="no scale covering these occurrences"):
        cf.export_metrics(measured_project, units="mm")


def test_only_millimetres_are_supported(measured_project):
    with pytest.raises(ValueError, match="isn't supported"):
        cf.export_metrics(measured_project, units="inches")


# ---------------------------------------------------------------------------
# occurrences_matching
# ---------------------------------------------------------------------------


def store_flags(project_path, flags, run_name="screening",
                source_mask_hash=None):
    recipe = Recipe("metric", run_name, [annotate_flags()], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(project_path, run_id, recipe.hash,
                   [make_metric_row(occurrence_id, "organism", "annotate_flags",
                                    flag, unit="category",
                                    source_mask_hash=source_mask_hash)
                    for occurrence_id, flag in flags.items()])


def test_stored_labels_can_be_selected_on_by_bare_metric_name(metadata_project):
    """
    The run and part prefixes are added for you, so a screening pass's usable
    crops are {"annotate_flags": "usable"}.
    """
    store_flags(metadata_project, {"specimen0": "usable", "specimen1": "cut_off",
                                   "specimen2": "usable"})
    assert occurrences_matching(metadata_project, "screening",
                                {"annotate_flags": "usable"}) == ["specimen0",
                                                                  "specimen2"]


def test_a_mistyped_metric_name_raises_once_there_is_data(metadata_project):
    """
    Because the usual thing to do with the answer is define a subset from it,
    and a subset that is silently empty looks exactly like a review pass nobody
    has done yet.
    """
    store_flags(metadata_project, {"specimen0": "usable"})
    with pytest.raises(KeyError, match="rule column"):
        occurrences_matching(metadata_project, "screening",
                             {"annotate_flag": "usable"})


def test_a_run_nobody_has_done_yet_selects_none_and_says_so(metadata_project,
                                                            caplog):
    """
    The one empty case that ISN'T a typo. There is nothing to check a rule
    against, so the typo guard can't apply -- it applies from the first
    recorded value onward.
    """
    with caplog.at_level("WARNING"):
        assert occurrences_matching(metadata_project, "screening",
                                    {"annotate_flags": "usable"}) == []
    assert "nothing to match" in caplog.text


def test_labels_survive_a_resegmentation_by_default(metadata_project):
    """
    current_only is False here, against the grain of everything else that reads
    stored values: a label like "cut_off" describes the CROP and stays true
    whatever mask was on screen. Left at True, resegmenting would void a review
    session over a change the labels never depended on.
    """
    from critterframe.records import masks as mask_records
    from helpers.synthetic import blob_mask

    store_flags(metadata_project, {"specimen0": "usable"},
                source_mask_hash="the_mask_that_was_on_screen")
    mask_records.save_masks(metadata_project, [
        mask_records.make_mask_row("specimen0", blob_mask(),
                                   recipe_hash="brand_new")])

    assert occurrences_matching(metadata_project, "screening",
                                {"annotate_flags": "usable"}) == ["specimen0"]
    assert occurrences_matching(metadata_project, "screening",
                                {"annotate_flags": "usable"},
                                current_only=True) == []


def test_numeric_labels_match_as_stored(metadata_project):
    """Values are compared as they were stored, without coercion."""
    recipe = Recipe("metric", "qc", [annotate_flags()], part="organism")
    run_id = start_run(metadata_project, recipe)
    append_metrics(metadata_project, run_id, recipe.hash, [
        make_metric_row("specimen0", "organism", "grade", 3),
        make_metric_row("specimen1", "organism", "grade", np.int64(4)),
    ])
    assert occurrences_matching(metadata_project, "qc", {"grade": 4}) == ["specimen1"]
