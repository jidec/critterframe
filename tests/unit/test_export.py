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

The last section is about the manifest, which exists because none of the above
survives into a CSV: the reshape drops every recipe hash and run id it read.
"""

import json

import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths as cf_paths
from critterframe.export import (
    _apply_filters,
    _to_millimetres,
    column_name,
    metric_units,
    metrics_wide,
    occurrences_matching,
)
from critterframe.metrics.annotation import annotate_flags
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.occurrences import ID_COL, load_occurrences
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
# _to_millimetres
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
    converted = _to_millimetres(wide_frame(), UNITS,
                               pd.Series({"a": 10.0, "b": 20.0}))
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


# ---------------------------------------------------------------------------
# _apply_filters
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
    assert _apply_filters(filterable(), {"length": condition})[ID_COL].tolist() == expected


def test_membership_filters_take_a_container():
    kept = _apply_filters(filterable(), {"flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["a", "c", "d"]
    excluded = _apply_filters(filterable(), {"flag": ("not in", ["usable"])})
    assert excluded[ID_COL].tolist() == ["b"]


def test_a_callable_expresses_what_the_shorthand_cannot():
    kept = _apply_filters(filterable(),
                         {"length": lambda series: series.between(20, 80)})
    assert kept[ID_COL].tolist() == ["b"]


def test_conditions_are_anded_together():
    kept = _apply_filters(filterable(),
                         {"length": (">", 20), "flag": ("in", ["usable"])})
    assert kept[ID_COL].tolist() == ["c"]


@pytest.mark.parametrize("condition", [(">", 0), ("<", 1000), ("!=", 1),
                                       ("not in", ["x"])])
def test_a_missing_value_never_passes(condition):
    """
    "This metric wasn't measured" must not quietly count as passing a !=
    test -- an unmeasured occurrence is not a verified-good one.
    """
    assert "d" not in _apply_filters(filterable(),
                                    {"length": condition})[ID_COL].tolist()


def test_filtering_on_a_column_that_is_not_there_raises():
    """A typo should be loud, not silently hand back an empty export."""
    with pytest.raises(KeyError, match="filter column"):
        _apply_filters(filterable(), {"lenght": (">", 1)})


def test_an_unsupported_operator_raises():
    with pytest.raises(ValueError, match="unsupported filter op"):
        _apply_filters(filterable(), {"length": ("~=", 1)})


# ---------------------------------------------------------------------------
# export_metrics
# ---------------------------------------------------------------------------


def test_export_defaults_to_a_uniquely_named_file_under_exports(measured_project):
    """
    No path named: still written, not lost, and never collides with a
    previous unnamed export.
    """
    exports = cf_paths.exports_dir(measured_project)
    assert not list(exports.glob("*.csv"))

    first = cf.export_metrics(measured_project)
    second = cf.export_metrics(measured_project)

    written = list(exports.glob("*.csv"))
    assert len(written) == 2
    assert len(pd.read_csv(written[0])) == len(first)
    assert len(pd.read_csv(written[1])) == len(second)


def test_path_false_returns_without_writing(measured_project, tmp_path):
    exports = cf_paths.exports_dir(measured_project)
    before = set(exports.glob("*.csv")) if exports.exists() else set()

    exported = cf.export_metrics(measured_project, path=False)

    after = set(exports.glob("*.csv")) if exports.exists() else set()
    assert after == before

    destination = tmp_path / "traits.csv"
    written = cf.export_metrics(measured_project, destination)
    assert len(pd.read_csv(destination)) == len(written) == len(exported)


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


# ---------------------------------------------------------------------------
# The manifest: what an export says about itself
# ---------------------------------------------------------------------------


def read_sidecar(path):
    """The manifest written beside one export."""
    return json.loads(cf_paths.export_sidecar_path(path).read_text(encoding="utf-8"))


def test_an_export_names_the_runs_and_recipes_behind_it(measured_project, tmp_path):
    """
    The gap this closes: a CSV that leaves the project used to say nothing about
    where its numbers came from. Every run that contributed a value is named,
    with the recipe hash that produced it.
    """
    out = tmp_path / "traits.csv"
    exported = cf.export_metrics(measured_project, out)

    record = read_sidecar(out)
    assert record["occurrences"]["count"] == len(exported)
    assert [run["name"] for run in record["runs"]] == ["traits"]
    assert record["runs"][0]["kind"] == "metric"
    assert len(record["runs"][0]["recipe_hash"]) == 16
    assert record["runs"][0]["recipe"]["kind"] == "metric"


def test_every_exported_column_says_what_it_holds(measured_project, tmp_path):
    """
    A column name alone does not say whether a number is pixels, a fraction or
    a category, which is the whole reason export_units exists. The manifest
    carries it per column, for exactly the columns the export ended up with.
    """
    out = tmp_path / "traits.csv"
    exported = cf.export_metrics(measured_project, out)

    occurrence_columns = set(load_occurrences(measured_project).columns)
    columns = read_sidecar(out)["columns"]
    assert set(columns) == {c for c in exported.columns if c not in occurrence_columns}

    length = columns["traits__organism__body_length"]
    assert (length["run_name"], length["part"]) == ("traits", "organism")
    assert (length["metric_name"], length["unit"]) == ("body_length", "px")


def test_the_masks_the_numbers_came_from_are_named(measured_project, tmp_path):
    """
    A metric row's source_mask_hash is the identity of the segmentation beneath
    it, so listing the distinct ones is what makes "these numbers came from
    those masks" answerable from the CSV's own manifest.
    """
    from critterframe.records import masks as mask_records

    out = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, out)

    record = read_sidecar(out)["source_masks"]
    current = set(mask_records.current_derivation_hashes(measured_project).values())
    assert set(record["derivations"]) == current
    assert record["n_without_provenance"] == 0


def test_the_hash_covers_the_data_and_not_the_filename(measured_project, tmp_path):
    """
    Two writes of one table are one export. The identity is what was selected
    and what came out, so the filename and the timestamp are recorded but sit
    outside the hash -- the same reasoning that keeps a path out of a registered
    model's identity().
    """
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    cf.export_metrics(measured_project, first)
    cf.export_metrics(measured_project, second)

    one, two = read_sidecar(first), read_sidecar(second)
    assert one["export_hash"] == two["export_hash"]
    assert one["path"] != two["path"]


def test_a_different_selection_is_a_different_export(measured_project, tmp_path):
    """The hash has to move when the table does, or it identifies nothing."""
    everything = tmp_path / "all.csv"
    narrowed = tmp_path / "some.csv"
    cf.export_metrics(measured_project, everything)
    cf.export_metrics(measured_project, narrowed,
                      metric_names=["body_length"])

    assert read_sidecar(everything)["export_hash"] \
        != read_sidecar(narrowed)["export_hash"]


def test_resegmenting_moves_the_export_hash(measured_project, tmp_path):
    """
    The point of recording the derivations: re-measure off new masks and the
    export is a different export, even though the recipe, the occurrences and
    the column names are all unchanged.
    """
    from helpers.models import ThresholdModel

    before = tmp_path / "before.csv"
    cf.export_metrics(measured_project, before)

    cf.run_segments(measured_project, steps=[cf.segment(ThresholdModel(erode=3))],
                    run_name="tighter", visualize=False)
    # The exact recipe _measured_template used under "traits" (conftest.py) --
    # run_name is pinned to a recipe, so re-measuring after resegmenting has to
    # be this same recipe rather than a narrower stand-in.
    cf.run_metrics(measured_project, run_name="traits",
                   transforms=[cf.remove_appendages(), cf.orient()],
                   metrics=[cf.body_length(), cf.max_width(),
                            cf.mask_area(name="area_px", unit="px2"),
                            cf.mean_lightness(), cf.blur_variance(),
                            cf.bilateral_asymmetry(), cf.edge_fraction()],
                   visualize=False)

    after = tmp_path / "after.csv"
    cf.export_metrics(measured_project, after)

    assert read_sidecar(before)["source_masks"]["derivations"] \
        != read_sidecar(after)["source_masks"]["derivations"]
    assert read_sidecar(before)["export_hash"] != read_sidecar(after)["export_hash"]


def test_how_the_rows_were_chosen_is_recorded(measured_project, tmp_path):
    """
    Filtering is revisable precisely because nothing is deleted -- which is only
    useful if the threshold that was applied is still knowable a year later.
    """
    out = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, out, units=None, current_only=True,
                      filters={"traits__organism__body_length": (">", 5)})

    selection = read_sidecar(out)["selection"]
    assert selection["filters"] == {"traits__organism__body_length": [">", 5]}
    assert selection["current_only"] is True
    assert selection["units"] is None


def test_a_predicate_filter_is_recorded_by_name(measured_project, tmp_path):
    """
    A callable cannot be stored, and its repr carries a memory address that
    would make one export hash differently on every run. The name is what is
    kept, and the docstring says so -- two different lambdas sharing a name are
    indistinguishable here.
    """
    def not_tiny(series):
        return series > 5

    out = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, out,
                      filters={"traits__organism__body_length": not_tiny})

    recorded = read_sidecar(out)["selection"]["filters"]
    assert recorded["traits__organism__body_length"]["callable"].endswith("not_tiny")

    again = tmp_path / "again.csv"
    cf.export_metrics(measured_project, again,
                      filters={"traits__organism__body_length": not_tiny})
    assert read_sidecar(out)["export_hash"] == read_sidecar(again)["export_hash"]


def test_the_project_logs_exports_it_wrote_elsewhere(measured_project, tmp_path):
    """
    The sidecar travels with the file; the log stays behind. An export written
    outside the project is exactly the case where only the log can answer "what
    has this project handed out".
    """
    cf.export_metrics(measured_project, tmp_path / "traits.csv")
    cf.export_metrics(measured_project, path=False)          # returned, never written

    log = cf.load_exports(measured_project)
    assert len(log) == 2
    assert log["path"].iloc[0].endswith("traits.csv")
    assert log["path"].iloc[1] is None
    assert not cf_paths.export_sidecar_path(tmp_path / "nothing.csv").exists()


def test_no_manifest_is_written_when_it_is_not_wanted(measured_project, tmp_path):
    """manifest=False leaves both the sidecar and the log alone."""
    out = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, out, manifest=False)

    assert out.exists()
    assert not cf_paths.export_sidecar_path(out).exists()
    assert cf.load_exports(measured_project).empty


def test_a_project_that_has_exported_nothing_reads_as_empty(measured_project):
    """No log file is 'nothing yet', not an error."""
    assert cf.load_exports(measured_project).empty
