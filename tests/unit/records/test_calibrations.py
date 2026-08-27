"""
Calibrations: knowledge about the imaging system, keyed by a scope, resolved
rather than copied.

Two properties carry the design, and both are tested here rather than in
`calibrations/scale.py`, because neither is about scale at all.

**The payload is opaque.** A scale is one number and a colour correction is a
method plus a matrix plus an offset plus an illuminant. Flattening both into a
`value` column would have fitted the first and distorted the second, so this
layer stores a JSON dict it never interprets -- and the test for that is a
colour row going in and coming back out with its matrix intact.

**A scope is just an occurrence column,** and the narrowest one wins. That is
what makes "a target in this particular frame" override "the copy stand this
frame was shot on" without anybody hardcoding a hierarchy of column names.
"""

import pandas as pd
import pytest

from critterframe.records import calibrations as calibration_records
from critterframe.records.occurrences import ID_COL, save_occurrences
from helpers.compare import is_iso_utc

COLOUR = {"method": "rgb_affine",
          "matrix": [[1.02, 0, 0], [0, 0.98, 0], [0, 0, 1.05]],
          "offset": [0.01, 0.0, -0.01],
          "illuminant": "D65"}


@pytest.fixture
def scoped_project(tmp_path):
    """
    Six occurrences with three nested groupings: a device (2 values), a session
    (3 values), and their own ids (6 values). Exactly what resolution has to
    order by specificity.
    """
    save_occurrences(tmp_path, pd.DataFrame({
        ID_COL: [f"occ{index}" for index in range(6)],
        "device": ["boxA", "boxA", "boxA", "boxB", "boxB", "boxB"],
        "session": ["s1", "s1", "s2", "s2", "s3", "s3"],
    }))
    return tmp_path


def row(scope="device", scope_value="boxA", parameters=None, **kwargs):
    kwargs.setdefault("source", "declared")
    return calibration_records.make_calibration_row(
        "scale", scope, scope_value, parameters or {"px_per_mm": 10.0}, **kwargs)


# ---------------------------------------------------------------------------
# make_calibration_row
# ---------------------------------------------------------------------------


def test_a_row_has_exactly_the_table_columns():
    assert set(row()) == set(calibration_records.COLUMNS)


def test_the_key_fields_are_strings():
    """Storage compares keys by value, so a numeric device id must not differ
    from the same id written as text."""
    made = calibration_records.make_calibration_row(
        "scale", "event_id", 12835, {"px_per_mm": 1.0}, source="target")
    assert made["scope_value"] == "12835"


def test_parameters_must_be_a_dict():
    """
    Even a single-number calibration is stored as one, so that adding a second
    number later doesn't change the shape of the table.
    """
    with pytest.raises(TypeError, match="must be a dict"):
        calibration_records.make_calibration_row("scale", "device", "boxA",
                                                 11.8, source="declared")


def test_the_score_and_source_are_kept():
    """
    A measured calibration and an asserted one deserve different amounts of
    trust, and six months later nothing else records which this was.
    """
    made = row(source="target", score=0.94, measured_from="sheet_12.jpg")
    assert made["source"] == "target"
    assert made["score"] == 0.94
    assert made["measured_from"] == "sheet_12.jpg"
    assert is_iso_utc(made["created_at"])


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_a_re_measurement_supersedes_rather_than_accumulates(tmp_path):
    """
    Unlike a metric there is no result to keep the history of: nothing was ever
    stored in converted units, so a corrected calibration invalidates nothing.
    """
    calibration_records.save_calibrations(tmp_path, [row(parameters={"px_per_mm": 10.0})])
    calibration_records.save_calibrations(tmp_path, [row(parameters={"px_per_mm": 12.0})])

    stored = calibration_records.load_calibrations(tmp_path)
    assert len(stored) == 1
    assert stored["parameters"].iloc[0] == {"px_per_mm": 12.0}


def test_the_type_is_part_of_the_key(tmp_path):
    """A colour calibration on a device must not displace its scale."""
    calibration_records.save_calibrations(tmp_path, [
        row(),
        calibration_records.make_calibration_row("color", "device", "boxA",
                                                 COLOUR, source="color_checker"),
    ])
    assert len(calibration_records.load_calibrations(tmp_path)) == 2


def test_an_opaque_payload_survives_the_round_trip(tmp_path):
    """
    The record layer never interprets `parameters`, which is what lets a
    calibration type grow a field without a schema change.
    """
    calibration_records.save_calibrations(tmp_path, [
        calibration_records.make_calibration_row("color", "device", "boxA",
                                                 COLOUR, source="color_checker")])
    stored = calibration_records.load_calibrations(tmp_path,
                                                   calibration_type="color")
    assert stored["parameters"].iloc[0] == COLOUR


def test_an_uncalibrated_project_reads_as_empty(tmp_path):
    """
    A project measuring only shape traits or only relative colour never needs a
    calibration at all.
    """
    empty = calibration_records.load_calibrations(tmp_path)
    assert empty.empty
    assert "parameters" in empty.columns


def test_load_filters_by_type_and_scope(tmp_path):
    calibration_records.save_calibrations(tmp_path, [
        row(scope="device", scope_value="boxA"),
        row(scope="session", scope_value="s1"),
    ])
    assert len(calibration_records.load_calibrations(tmp_path, scope="session")) == 1
    assert len(calibration_records.load_calibrations(tmp_path,
                                                     calibration_type="color")) == 0


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_a_scope_must_be_a_column_the_project_has(scoped_project):
    """
    Checked against the parquet schema before reading, because pyarrow's own
    error talks about FieldRefs and says nothing about scopes.
    """
    with pytest.raises(KeyError, match="nothing to key a calibration on"):
        calibration_records.require_scope_column(scoped_project, "copy_stand")


def test_pending_values_are_the_ones_not_yet_measured(scoped_project):
    calibration_records.save_calibrations(scoped_project,
                                          [row(scope="session", scope_value="s1")])
    pending = calibration_records.pending_scope_values(scoped_project, "scale",
                                                       "session")
    assert pending == ["s2", "s3"]


def test_pending_values_are_empty_once_everything_is_measured(scoped_project):
    calibration_records.save_calibrations(scoped_project, [
        row(scope="session", scope_value=value) for value in ("s1", "s2", "s3")])
    assert calibration_records.pending_scope_values(scoped_project, "scale",
                                                    "session") == []


def test_pending_values_can_be_capped(scoped_project):
    assert len(calibration_records.pending_scope_values(
        scoped_project, "scale", "session", limit=2)) == 2


# ---------------------------------------------------------------------------
# resolve_for_occurrences
# ---------------------------------------------------------------------------


def test_a_device_calibration_reaches_every_occurrence_shot_on_it(scoped_project):
    calibration_records.save_calibrations(scoped_project, [row()])
    resolved = calibration_records.resolve_for_occurrences(scoped_project, "scale")

    assert resolved["occ0"] == {"px_per_mm": 10.0}
    assert resolved["occ3"] is None


def test_an_uncalibrated_occurrence_gets_nothing_rather_than_an_average(
        scoped_project):
    """
    Never a project-wide average, never the nearest session's value. A trait
    converted with a guessed calibration is indistinguishable in a CSV from one
    converted with a measured calibration.
    """
    calibration_records.save_calibrations(scoped_project, [row()])
    resolved = calibration_records.resolve_for_occurrences(scoped_project, "scale")
    assert resolved.isna().sum() == 3


def test_the_narrower_scope_wins(scoped_project, caplog):
    """
    A calibration measured per deployment says more about one night than one
    measured per device says about a season -- and the ranking comes from how
    many occurrences each scope value covers, so a project inventing its own
    scope gets sensible precedence for free.
    """
    calibration_records.save_calibrations(scoped_project, [
        row(scope="device", scope_value="boxA", parameters={"px_per_mm": 10.0}),
        row(scope="session", scope_value="s1", parameters={"px_per_mm": 11.0}),
    ])
    with caplog.at_level("WARNING"):
        resolved = calibration_records.resolve_for_occurrences(scoped_project,
                                                               "scale")

    assert resolved["occ0"] == {"px_per_mm": 11.0}     # session s1
    assert resolved["occ2"] == {"px_per_mm": 10.0}     # boxA, session s2
    assert "more specific and overrides" in caplog.text


def test_an_occurrence_scoped_row_beats_everything(scoped_project):
    """
    It describes that occurrence and nothing else, which is the most specific
    statement available -- "a target in this particular frame".
    """
    calibration_records.save_calibrations(scoped_project, [
        row(scope="device", scope_value="boxA", parameters={"px_per_mm": 10.0}),
        row(scope=ID_COL, scope_value="occ0", parameters={"px_per_mm": 42.0}),
    ])
    resolved = calibration_records.resolve_for_occurrences(scoped_project, "scale")

    assert resolved["occ0"] == {"px_per_mm": 42.0}
    assert resolved["occ1"] == {"px_per_mm": 10.0}


def test_rows_scoped_on_a_column_the_project_lacks_are_reported(scoped_project,
                                                                caplog):
    """
    They apply to nothing, and saying so is the difference between "this
    project isn't calibrated" and "your calibration named a column that isn't
    here".
    """
    calibration_records.save_calibrations(scoped_project,
                                          [row(scope="copy_stand", scope_value="A")])
    with caplog.at_level("WARNING"):
        resolved = calibration_records.resolve_for_occurrences(scoped_project,
                                                               "scale")
    assert resolved.isna().all()
    assert "the occurrence table doesn't have" in caplog.text


def test_resolution_can_be_narrowed_to_some_occurrences(scoped_project):
    calibration_records.save_calibrations(scoped_project, [row()])
    resolved = calibration_records.resolve_for_occurrences(
        scoped_project, "scale", occurrence_ids=["occ0", "occ1"])
    assert resolved.index.tolist() == ["occ0", "occ1"]


def test_resolving_an_uncalibrated_type_is_an_empty_series(scoped_project):
    calibration_records.save_calibrations(scoped_project, [row()])
    assert calibration_records.resolve_for_occurrences(scoped_project,
                                                       "color").empty


def test_two_types_resolve_independently(scoped_project):
    """
    Same scope machinery, arbitrary parameters, no schema change -- and the
    scale row is untouched, because the key includes the type.
    """
    calibration_records.save_calibrations(scoped_project, [
        row(),
        calibration_records.make_calibration_row("color", "device", "boxA",
                                                 COLOUR, source="color_checker"),
    ])
    scale = calibration_records.resolve_for_occurrences(scoped_project, "scale")
    colour = calibration_records.resolve_for_occurrences(scoped_project, "color")

    assert scale["occ0"] == {"px_per_mm": 10.0}
    assert colour["occ0"]["illuminant"] == "D65"
    assert colour["occ3"] is None
