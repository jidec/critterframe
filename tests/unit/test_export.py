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

import pandas as pd
import pytest

import critterframe as cf
from critterframe.project import paths as cf_paths
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.occurrences import ID_COL, load_occurrences
from critterframe.records.runs import start_run
from critterframe.core.recipes import Recipe
from critterframe.metrics.dimensions import body_length


def store_values(
    project_path, values, run_name="traits", part="organism", metric_name="body_length", unit="px"
):
    """Append {occurrence_id: value} under a fresh run of `run_name`."""
    recipe = Recipe("metric", run_name, [body_length()], part=part)
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [
            make_metric_row(occurrence_id, part, metric_name, value, unit=unit)
            for occurrence_id, value in values.items()
        ],
    )


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
    exported = cf.export_metrics(measured_project, occurrence_columns=["device", "species"])
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
    filtered = cf.export_metrics(measured_project, filters={column: (">", everything[column].median())})

    assert 0 < len(filtered) < len(everything)
    assert len(cf.export_metrics(measured_project)) == len(everything)


QC_AREA = "qc__organism__mask_area"


@pytest.fixture
def screened_project(measured_project):
    """The measured project plus a 'qc' run, the kind a filter reads and an analysis doesn't."""
    cf.run_metrics(
        measured_project, run_name="qc", visualize=False, metrics=[cf.mask_area(), cf.mask_fraction()]
    )
    return measured_project


def qc_cutoff(project_path, column=QC_AREA, **kwargs):
    """A cutoff some occurrences pass and some don't."""
    return cf.export_metrics(project_path, path=False, manifest=False, **kwargs)[column].median()


@pytest.mark.parametrize(
    "selection",
    [
        {"run_names": ["traits"]},
        {"metric_names": ["body_length"]},
    ],
)
def test_a_filter_reads_a_column_the_selection_left_out(screened_project, selection):
    """What decides the rows and what goes in the file are two selections."""
    unfiltered = cf.export_metrics(screened_project, path=False, **selection)
    filtered = cf.export_metrics(
        screened_project, path=False, **selection, filters={QC_AREA: (">", qc_cutoff(screened_project))}
    )

    assert 0 < len(filtered) < len(unfiltered)
    assert filtered.columns.tolist() == unfiltered.columns.tolist()


def test_a_selected_column_that_is_filtered_on_is_still_exported(screened_project):
    column = "traits__organism__body_length"
    filtered = cf.export_metrics(
        screened_project,
        path=False,
        run_names=["traits"],
        filters={column: (">", 0), QC_AREA: (">", qc_cutoff(screened_project))},
    )

    assert column in filtered.columns
    assert not [c for c in filtered.columns if c.startswith("qc__")]


def test_an_unnarrowed_export_keeps_its_filter_columns(screened_project):
    """Leaving a column out is the selection's doing, never the filter's."""
    filtered = cf.export_metrics(
        screened_project, path=False, filters={QC_AREA: (">", qc_cutoff(screened_project))}
    )
    assert QC_AREA in filtered.columns


def test_a_filter_only_run_is_named_in_the_manifest(screened_project, tmp_path):
    """Its columns aren't exported, but it decided the rows."""
    out = tmp_path / "traits.csv"
    cf.export_metrics(
        screened_project, out, run_names=["traits"], filters={QC_AREA: (">", qc_cutoff(screened_project))}
    )

    record = json.loads(cf_paths.export_sidecar_path(out).read_text(encoding="utf-8"))
    assert sorted(run["name"] for run in record["runs"]) == ["qc", "traits"]
    assert not [c for c in record["columns"] if c.startswith("qc__")]
    assert QC_AREA in record["selection"]["filters"]


def test_a_filter_on_an_unselected_column_can_be_written_in_millimetres(screened_project):
    cf.declare_scale(screened_project, 4.0, scope="device", scope_value="boxA")
    converted = f"{QC_AREA}_mm2"
    cutoff = qc_cutoff(screened_project, column=converted, units="mm")
    unfiltered = cf.export_metrics(screened_project, path=False, units="mm", run_names=["traits"])
    filtered = cf.export_metrics(
        screened_project, path=False, units="mm", run_names=["traits"], filters={converted: (">", cutoff)}
    )

    assert 0 < len(filtered) < len(unfiltered)
    assert filtered.columns.tolist() == unfiltered.columns.tolist()


def test_a_filter_column_no_run_holds_still_raises(screened_project):
    with pytest.raises(KeyError, match="not in the export"):
        cf.export_metrics(
            screened_project, path=False, run_names=["traits"], filters={"qc__organism__nope": (">", 0)}
        )


def test_a_filter_only_value_does_not_count_as_measured(metadata_project):
    """drop_empty is judged on the selected columns; a QC score alone isn't a trait."""
    first, second, third = load_occurrences(metadata_project)[ID_COL].tolist()[:3]
    store_values(metadata_project, {first: 10.0, second: 20.0})
    store_values(
        metadata_project,
        {first: 1.0, second: 1.0, third: 1.0},
        run_name="qc",
        metric_name="score",
        unit="score",
    )

    exported = cf.export_metrics(
        metadata_project, path=False, run_names=["traits"], filters={"qc__organism__score": (">", 0)}
    )
    assert sorted(exported[ID_COL]) == sorted([first, second])


# ---------------------------------------------------------------------------
# part_filters
# ---------------------------------------------------------------------------

HEAD = "traits__head__body_length"
ABDOMEN = "traits__abdomen__body_length"


@pytest.fixture
def two_part_project(metadata_project):
    """Three occurrences: one with both parts, one with a short abdomen, one with no head."""
    first, second, third = load_occurrences(metadata_project)[ID_COL].tolist()[:3]
    store_values(metadata_project, {first: 10.0, second: 20.0}, part="head")
    store_values(metadata_project, {first: 50.0, second: 5.0, third: 60.0}, part="abdomen")
    return metadata_project, (first, second, third)


def by_id(frame):
    return frame.set_index(ID_COL)


def test_a_part_failing_its_filters_is_emptied_and_the_row_kept(two_part_project):
    project, (first, second, third) = two_part_project
    exported = by_id(
        cf.export_metrics(project, path=False, manifest=False, part_filters={"abdomen": {ABDOMEN: (">", 10)}})
    )

    assert exported.loc[second, HEAD] == 20.0 and pd.isna(exported.loc[second, ABDOMEN])
    assert exported.loc[first, ABDOMEN] == 50.0
    # never measured is not a failure of the other part
    assert exported.loc[third, ABDOMEN] == 60.0 and pd.isna(exported.loc[third, HEAD])


def test_a_row_filter_would_have_dropped_the_whole_occurrence(two_part_project):
    project, (first, second, third) = two_part_project
    exported = cf.export_metrics(project, path=False, manifest=False, filters={ABDOMEN: (">", 10)})
    assert second not in set(exported[ID_COL])


def test_a_row_with_no_part_left_is_dropped_unless_empty_rows_are_kept(two_part_project):
    project, (first, second, third) = two_part_project
    rules = {"abdomen": {ABDOMEN: (">", 55)}, "head": {HEAD: (">", 15)}}

    exported = cf.export_metrics(project, path=False, manifest=False, part_filters=rules)
    assert sorted(exported[ID_COL]) == sorted([second, third])

    everyone = cf.export_metrics(project, path=False, manifest=False, part_filters=rules, drop_empty=False)
    assert first in set(everyone[ID_COL])
    assert by_id(everyone).loc[first, [HEAD, ABDOMEN]].isna().all()


def test_every_part_is_judged_before_any_is_emptied(two_part_project):
    """The head's rule reads the abdomen's column, which the abdomen's own rule empties."""
    project, (first, second, third) = two_part_project
    exported = by_id(
        cf.export_metrics(
            project,
            path=False,
            manifest=False,
            part_filters={"abdomen": {ABDOMEN: (">", 10)}, "head": {ABDOMEN: ("<", 10)}},
        )
    )

    assert exported.loc[second, HEAD] == 20.0 and pd.isna(exported.loc[second, ABDOMEN])
    assert pd.isna(exported.loc[first, HEAD]) and exported.loc[first, ABDOMEN] == 50.0


def test_row_filters_run_before_part_filters(two_part_project):
    project, (first, second, third) = two_part_project
    exported = cf.export_metrics(
        project,
        path=False,
        manifest=False,
        filters={ABDOMEN: ("<", 55)},
        part_filters={"abdomen": {ABDOMEN: (">", 10)}},
    )
    assert sorted(exported[ID_COL]) == sorted([first, second])
    assert pd.isna(by_id(exported).loc[second, ABDOMEN])


def test_a_part_filter_reads_a_column_the_selection_left_out(two_part_project):
    project, (first, second, third) = two_part_project
    store_values(
        project,
        {first: 0.9, second: 0.1, third: 0.9},
        run_name="qc",
        part="abdomen",
        metric_name="score",
        unit="score",
    )
    exported = by_id(
        cf.export_metrics(
            project,
            path=False,
            manifest=False,
            run_names=["traits"],
            part_filters={"abdomen": {"qc__abdomen__score": (">", 0.5)}},
        )
    )

    assert {HEAD, ABDOMEN} <= set(exported.columns)
    assert not [column for column in exported.columns if column.startswith("qc__")]
    assert pd.isna(exported.loc[second, ABDOMEN]) and exported.loc[first, ABDOMEN] == 50.0


def test_a_part_with_nothing_exported_cannot_be_filtered(two_part_project):
    project, _ids = two_part_project
    with pytest.raises(ValueError, match="no exported column"):
        cf.export_metrics(project, path=False, manifest=False, part_filters={"thorax": {ABDOMEN: (">", 10)}})
    with pytest.raises(KeyError, match="not in the export"):
        cf.export_metrics(
            project,
            path=False,
            manifest=False,
            part_filters={"abdomen": {"traits__abdomen__nope": (">", 10)}},
        )


def test_part_filters_are_in_the_manifest_with_what_they_did(two_part_project, tmp_path):
    project, (first, second, third) = two_part_project
    out = tmp_path / "parts.csv"
    cf.export_metrics(project, out, part_filters={"abdomen": {ABDOMEN: (">", 10)}})
    record = json.loads(cf_paths.export_sidecar_path(out).read_text(encoding="utf-8"))

    assert record["selection"]["part_filters"] == {"abdomen": {ABDOMEN: [">", 10]}}
    assert record["part_counts"] == {"abdomen": {"kept": 2, "filtered_out": 1, "not_measured": 0}}
    # the emptied abdomen value is not among the values the file is said to hold
    assert record["source_masks"]["n_values"] == 4


def test_an_export_without_part_filters_records_nothing_about_them(two_part_project, tmp_path):
    """So every export hash written before part filters existed still holds."""
    project, _ids = two_part_project
    plain, filtered, moved = (tmp_path / name for name in ("a.csv", "b.csv", "c.csv"))
    cf.export_metrics(project, plain)
    cf.export_metrics(project, filtered, part_filters={"abdomen": {ABDOMEN: (">", 10)}})
    cf.export_metrics(project, moved, part_filters={"abdomen": {ABDOMEN: (">", 55)}})
    records = [
        json.loads(cf_paths.export_sidecar_path(path).read_text(encoding="utf-8"))
        for path in (plain, filtered, moved)
    ]

    assert "part_counts" not in records[0] and "part_filters" not in records[0]["selection"]
    assert len({record["export_hash"] for record in records}) == 3


def test_a_part_filter_can_be_written_in_millimetres(screened_project):
    cf.declare_scale(screened_project, 4.0, scope="device", scope_value="boxA")
    converted = f"{QC_AREA}_mm2"
    cutoff = qc_cutoff(screened_project, column=converted, units="mm")
    by_row = cf.export_metrics(
        screened_project,
        path=False,
        units="mm",
        manifest=False,
        run_names=["traits"],
        filters={converted: (">", cutoff)},
    )
    by_part = cf.export_metrics(
        screened_project,
        path=False,
        units="mm",
        manifest=False,
        run_names=["traits"],
        part_filters={"organism": {converted: (">", cutoff)}},
    )

    # one part exported, so emptying it is dropping the row
    assert sorted(by_part[ID_COL]) == sorted(by_row[ID_COL])
    assert by_part.columns.tolist() == by_row.columns.tolist()


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
    cf.export_metrics(measured_project, narrowed, metric_names=["body_length"])

    assert read_sidecar(everything)["export_hash"] != read_sidecar(narrowed)["export_hash"]


def test_resegmenting_moves_the_export_hash(measured_project, tmp_path):
    """
    The point of recording the derivations: re-measure off new masks and the
    export is a different export, even though the recipe, the occurrences and
    the column names are all unchanged.
    """
    from helpers.models import ThresholdModel

    before = tmp_path / "before.csv"
    cf.export_metrics(measured_project, before)

    cf.run_segments(
        measured_project, steps=[cf.segment(ThresholdModel(erode=3))], run_name="tighter", visualize=False
    )
    # The exact recipe _measured_template used under "traits" (conftest.py) --
    # run_name is pinned to a recipe, so re-measuring after resegmenting has to
    # be this same recipe rather than a narrower stand-in.
    cf.run_metrics(
        measured_project,
        run_name="traits",
        transforms=[cf.remove_appendages(), cf.orient()],
        metrics=[
            cf.body_length(),
            cf.max_width(),
            cf.mask_area(name="area_px", unit="px2"),
            cf.mean_lightness(),
            cf.blur_variance(),
            cf.bilateral_asymmetry(),
            cf.edge_fraction(),
        ],
        visualize=False,
    )

    after = tmp_path / "after.csv"
    cf.export_metrics(measured_project, after)

    assert (
        read_sidecar(before)["source_masks"]["derivations"]
        != read_sidecar(after)["source_masks"]["derivations"]
    )
    assert read_sidecar(before)["export_hash"] != read_sidecar(after)["export_hash"]


def test_how_the_rows_were_chosen_is_recorded(measured_project, tmp_path):
    """
    Filtering is revisable precisely because nothing is deleted -- which is only
    useful if the threshold that was applied is still knowable a year later.
    """
    out = tmp_path / "traits.csv"
    cf.export_metrics(
        measured_project,
        out,
        units=None,
        current_only=True,
        filters={"traits__organism__body_length": (">", 5)},
    )

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
    cf.export_metrics(measured_project, out, filters={"traits__organism__body_length": not_tiny})

    recorded = read_sidecar(out)["selection"]["filters"]
    assert recorded["traits__organism__body_length"]["callable"].endswith("not_tiny")

    again = tmp_path / "again.csv"
    cf.export_metrics(measured_project, again, filters={"traits__organism__body_length": not_tiny})
    assert read_sidecar(out)["export_hash"] == read_sidecar(again)["export_hash"]


def test_the_project_logs_exports_it_wrote_elsewhere(measured_project, tmp_path):
    """
    The sidecar travels with the file; the log stays behind. An export written
    outside the project is exactly the case where only the log can answer "what
    has this project handed out".
    """
    cf.export_metrics(measured_project, tmp_path / "traits.csv")
    cf.export_metrics(measured_project, path=False)  # returned, never written

    log = cf.load_exports(measured_project)
    assert len(log) == 2
    assert log["path"].iloc[0].endswith("traits.csv")
    assert log["path"].iloc[1] is None
    assert not cf_paths.export_sidecar_path(tmp_path / "nothing.csv").exists()


def test_a_rename_relabels_and_changes_nothing_else(measured_project, tmp_path):
    """Same values, same column order, in the frame and in the file."""
    column = "traits__organism__body_length"
    plain = cf.export_metrics(measured_project, path=False)
    out = tmp_path / "traits.csv"
    renamed = cf.export_metrics(measured_project, out, rename={column: "length"})

    assert renamed.columns.tolist() == ["length" if c == column else c for c in plain.columns]
    assert renamed["length"].tolist() == plain[column].tolist()
    assert pd.read_csv(out).columns.tolist() == renamed.columns.tolist()


def test_a_renamed_column_still_says_what_it_holds(measured_project, tmp_path):
    """
    The manifest is keyed by the header the CSV actually carries, so a short
    name does not cost the column its run, part, metric and unit.
    """
    column = "traits__organism__body_length"
    out = tmp_path / "traits.csv"
    cf.export_metrics(measured_project, out, rename={column: "length"})

    record = read_sidecar(out)
    assert column not in record["columns"]
    length = record["columns"]["length"]
    assert length["column"] == column
    assert (length["run_name"], length["part"]) == ("traits", "organism")
    assert (length["metric_name"], length["unit"]) == ("body_length", "px")
    assert record["selection"]["rename"] == {column: "length"}
    # a column left alone carries no second name
    assert "column" not in record["columns"]["traits__organism__area_px"]


def test_a_rename_names_the_converted_column(measured_project, tmp_path):
    """Under units='mm' the name to rename is the one filters use, with its suffix."""
    cf.declare_scale(measured_project, 4.0, scope="device", scope_value="boxA")
    converted = "traits__organism__body_length_mm"
    out = tmp_path / "traits.csv"
    exported = cf.export_metrics(measured_project, out, units="mm", rename={converted: "length_mm"})

    assert "length_mm" in exported.columns
    length = read_sidecar(out)["columns"]["length_mm"]
    assert (length["column"], length["unit"], length["source_unit"]) == (converted, "mm", "px")


def test_filters_take_default_names_alongside_a_rename(measured_project):
    column = "traits__organism__body_length"
    everything = cf.export_metrics(measured_project, path=False)
    filtered = cf.export_metrics(
        measured_project,
        path=False,
        rename={column: "length"},
        filters={column: (">", everything[column].median())},
    )

    assert 0 < len(filtered) < len(everything)
    assert "length" in filtered.columns


@pytest.mark.parametrize(
    "rename, error, match",
    [
        ({"traits__organism__nope": "x"}, KeyError, "not in the export"),
        ({"traits__organism__body_length": "traits__organism__area_px"}, ValueError, "already columns"),
        (
            {"traits__organism__body_length": "x", "traits__organism__area_px": "x"},
            ValueError,
            "more than one column",
        ),
        ({ID_COL: "id"}, ValueError, "can't be renamed"),
    ],
)
def test_a_rename_that_cannot_be_honoured_raises(measured_project, rename, error, match):
    with pytest.raises(error, match=match):
        cf.export_metrics(measured_project, path=False, rename=rename)


def test_a_rename_is_in_the_hash_only_when_given(measured_project, tmp_path):
    """An export made before the option existed keeps its hash; a relabelled table is a different one."""
    omitted, empty, renamed = (tmp_path / name for name in ("omitted.csv", "empty.csv", "renamed.csv"))
    cf.export_metrics(measured_project, omitted)
    cf.export_metrics(measured_project, empty, rename={})
    cf.export_metrics(measured_project, renamed, rename={"traits__organism__body_length": "length"})

    assert "rename" not in read_sidecar(omitted)["selection"]
    assert read_sidecar(omitted)["export_hash"] == read_sidecar(empty)["export_hash"]
    assert read_sidecar(omitted)["export_hash"] != read_sidecar(renamed)["export_hash"]


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


def test_an_export_inside_the_project_is_logged_relative_to_it(measured_project, tmp_path):
    """So a copied project's log still names its own files; one written elsewhere stays absolute."""
    cf.export_metrics(measured_project)  # the default: exports/
    cf.export_metrics(measured_project, tmp_path / "elsewhere.csv")

    inside, outside = cf.load_exports(measured_project)["path"]
    assert inside.startswith("exports/") and inside.endswith(".csv")
    assert cf_paths.is_absolute_anywhere(outside)
