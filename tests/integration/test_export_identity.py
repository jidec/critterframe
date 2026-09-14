"""
An export says what it is, and stops saying it when the data moves.

Everything the package knows about where a number came from -- the recipe hash
that produced it, the run that recorded it, the mask it was measured from -- is
read inside `export_metrics` and then dropped by the long-to-wide reshape. A
CSV is a table of bare numbers, and the project it came from kept no note that
it had ever been written.

So every export now writes a manifest twice over: beside the file, because a
CSV handed to a collaborator has to carry its own identity, and into the
project's exports log, because an export written somewhere else entirely is
exactly the case where only the project can answer what it handed out.

The rule the `export_hash` follows: it covers what was selected and what came
out, and nothing else. Same table, two filenames, one export -- the same
reasoning that keeps a path out of a registered model's `identity()`. Move the
masks underneath and it must move, even though the recipe, the occurrences and
the column names are all untouched.
"""

import json

import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.records import masks as mask_records
from critterframe.records import runs as run_records
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

LENGTH = "traits__organism__body_length"


def sidecar(path):
    """The manifest written beside one export."""
    return json.loads(paths.export_sidecar_path(path).read_text(encoding="utf-8"))


def test_a_csv_carries_the_whole_chain_that_produced_it(measured_project,
                                                        tmp_path):
    """
    Ingest through export, then read the provenance back off the file alone:
    which occurrences, which runs, which recipes, which masks, in what units.
    """
    out = tmp_path / "traits.csv"
    exported = cf.export_metrics(measured_project, out)
    record = sidecar(out)

    assert record["occurrences"]["count"] == len(exported)

    stored = run_records.load_runs(measured_project, name="traits")
    assert [run["recipe_hash"] for run in record["runs"]] \
        == [stored["recipe_hash"].iloc[0]]

    assert set(record["source_masks"]["derivations"]) == set(
        mask_records.current_derivation_hashes(measured_project).values())

    assert record["columns"][LENGTH]["unit"] == "px"
    assert record["project"].endswith("project")


def test_resegmenting_makes_the_next_export_a_different_export(measured_project,
                                                               tmp_path):
    """
    The staleness rule, seen from the far end. A resegmentation moves the
    identity of every mask, so the values measured from the new ones are a
    different body of data even though nothing about the recipe changed -- and
    two CSVs that look alike column for column must not claim to be the same
    export.
    """
    before = tmp_path / "before.csv"
    cf.export_metrics(measured_project, before)

    cf.run_segments(measured_project, run_name="tighter",
                    steps=[cf.segment(ThresholdModel(erode=3))], visualize=False)
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

    assert sidecar(before)["source_masks"]["derivations"] \
        != sidecar(after)["source_masks"]["derivations"]
    assert sidecar(before)["export_hash"] != sidecar(after)["export_hash"]


def test_the_project_keeps_a_history_of_what_it_handed_out(measured_project,
                                                           tmp_path):
    """
    The log is the half that stays behind. It records the export whose file
    went elsewhere and the one that was never written at all, which is the
    whole reason it isn't just the sidecar.
    """
    cf.export_metrics(measured_project, tmp_path / "for_a_colleague.csv")
    cf.export_metrics(measured_project, filters={LENGTH: (">", 5)}, path=False)

    log = cf.load_exports(measured_project)
    assert len(log) == 2
    assert log["path"].iloc[0].endswith("for_a_colleague.csv")
    assert log["path"].iloc[1] is None
    assert log["export_hash"].nunique() == 2


def test_a_millimetre_export_records_the_calibration_behind_it(measured_project,
                                                               tmp_path):
    """
    A converted column is only as good as the calibration under it, and the
    px_per_mm riding along in the CSV is one number with no account of where it
    came from or how many occurrences it failed to reach.
    """
    cf.declare_scale(measured_project, 4.0, scope="device", scope_value="boxA")

    out = tmp_path / "mm.csv"
    cf.export_metrics(measured_project, out, units="mm")
    calibration = sidecar(out)["calibration"]

    assert calibration["type"] == "scale"
    assert calibration["scopes"] == {"device": 1}
    assert calibration["sources"] == {"declared": 1}
    # boxA is calibrated, boxB is not, and the manifest says so rather than
    # leaving a reader to notice the empty column.
    assert calibration["covered"] == 4
    assert calibration["uncovered"] == 4
    assert sidecar(out)["columns"][f"{LENGTH}_mm"]["source_unit"] == "px"
