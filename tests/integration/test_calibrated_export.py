"""
Millimetres, filters, and the two rules that keep both revisable.

**A calibration is applied late, never baked in.** Traits are measured in pixels
and stay that way; `export_metrics(units="mm")` converts at the last moment. The
alternative -- measuring in millimetres -- would put the calibration inside the
recipe hash, which would make a corrected calibration a different recipe and
invalidate every stored trait in the project. Measuring stays in pixels forever;
a corrected calibration costs one re-export.

**Filtering happens at export only, and deletes nothing.** A threshold is a
judgement about degree that a project must be able to revise without recomputing
anything, so it narrows the output and leaves the data alone.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.records.occurrences import ID_COL

pytestmark = pytest.mark.slow

SPECIMENS = 8
LENGTH = "traits__organism__body_length"
LENGTH_MM = "traits__organism__body_length_mm"
AREA_MM2 = "traits__organism__area_px_mm2"
LIGHTNESS = "traits__organism__mean_lightness"


@pytest.fixture
def calibrated(measured_project):
    """
    boxA's rig is calibrated at 4 px/mm; boxB's is not. specimen0 was shot with
    a target in frame, so it carries its own 8 px/mm.
    """
    cf.declare_scale(measured_project, 4.0, scope="device", scope_value="boxA")
    cf.declare_scale(measured_project, 8.0, scope=ID_COL, scope_value="specimen0")
    return measured_project


def test_the_narrowest_calibration_wins(calibrated):
    """
    A scale measured on one frame is more specific than one measured for the
    rig it was shot on -- and an uncalibrated occurrence stays NaN rather than
    borrowing its neighbours'.
    """
    scale = cf.scale_for_occurrences(calibrated)

    assert scale["specimen0"] == 8.0
    assert scale["specimen2"] == 4.0            # boxA
    assert np.isnan(scale["specimen1"])         # boxB, uncalibrated


def test_lengths_and_areas_convert_by_the_right_power(calibrated):
    pixels = cf.export_metrics(calibrated).set_index(ID_COL)
    millimetres = cf.export_metrics(calibrated, units="mm").set_index(ID_COL)

    assert millimetres.loc["specimen2", LENGTH_MM] == \
        pytest.approx(pixels.loc["specimen2", LENGTH] / 4.0)
    assert millimetres.loc["specimen2", AREA_MM2] == \
        pytest.approx(pixels.loc["specimen2", "traits__organism__area_px"] / 16.0)


def test_a_fraction_is_left_alone(calibrated):
    """There is no length in it to convert."""
    pixels = cf.export_metrics(calibrated).set_index(ID_COL)
    millimetres = cf.export_metrics(calibrated, units="mm").set_index(ID_COL)
    assert millimetres.loc["specimen2", LIGHTNESS] == pixels.loc["specimen2", LIGHTNESS]


def test_an_uncalibrated_occurrence_is_nan_never_raw_pixels(calibrated):
    """
    The same number meaning something entirely different in the same column is
    the failure this prevents -- and it would be invisible in a CSV.
    """
    millimetres = cf.export_metrics(calibrated, units="mm").set_index(ID_COL)
    assert np.isnan(millimetres.loc["specimen1", LENGTH_MM])
    assert millimetres[LENGTH_MM].notna().sum() == 4       # boxA only


def test_an_uncalibrated_occurrence_is_still_exported(calibrated):
    """
    Converted AFTER drop_empty, so "has any measurement" is judged on the
    stored values: an occurrence measured perfectly well but lacking a
    calibration appears with empty millimetre columns rather than vanishing.
    """
    assert len(cf.export_metrics(calibrated, units="mm")) == SPECIMENS


def test_the_scale_used_rides_along_in_the_export(calibrated):
    millimetres = cf.export_metrics(calibrated, units="mm").set_index(ID_COL)
    assert millimetres.loc["specimen0", "px_per_mm"] == 8.0


def test_nothing_stored_changes_when_units_do(calibrated):
    """
    Applied late means exactly this: the project holds the same pixel values
    before and after a millimetre export, so a corrected calibration costs one
    re-export and nothing else.
    """
    before = cf.export_metrics(calibrated)[LENGTH].tolist()
    cf.export_metrics(calibrated, units="mm")
    assert cf.export_metrics(calibrated)[LENGTH].tolist() == before


def test_a_corrected_calibration_changes_only_the_export(calibrated):
    cf.declare_scale(calibrated, 16.0, scope=ID_COL, scope_value="specimen0")
    millimetres = cf.export_metrics(calibrated, units="mm").set_index(ID_COL)

    assert millimetres.loc["specimen0", "px_per_mm"] == 16.0
    # The measurement it was divided into is untouched.
    assert cf.export_metrics(calibrated).set_index(ID_COL).loc[
        "specimen0", LENGTH] > 0


def test_a_filter_can_be_written_against_the_millimetre_column(calibrated):
    """
    Converted before filters, so a threshold in real units is expressible --
    which is the point of having real units at all.
    """
    filtered = cf.export_metrics(calibrated, units="mm",
                                 filters={LENGTH_MM: (">", 0.0)})
    assert len(filtered) == 4        # the calibrated ones; NaN never passes


def test_filtering_narrows_the_export_and_keeps_the_data(calibrated):
    everything = cf.export_metrics(calibrated)
    threshold = everything[LENGTH].median()

    filtered = cf.export_metrics(calibrated, filters={LENGTH: (">", threshold)})
    assert 0 < len(filtered) < SPECIMENS

    # Nothing was deleted: the same export without the filter is unchanged, and
    # so is the project's occurrence table.
    assert len(cf.export_metrics(calibrated)) == SPECIMENS
    assert len(cf.summarize(calibrated)["metrics"]["names"]) == 7


def test_a_subset_and_a_filter_compose(calibrated):
    cf.define_subset(calibrated, "boxA", column="device", values=["boxA"])
    filtered = cf.export_metrics(calibrated, subset="boxA",
                                 filters={LIGHTNESS: (">", 0.0)})
    assert len(filtered) == 4


def test_units_are_reported_for_what_the_column_now_holds(calibrated):
    """
    A unit stopped being merely descriptive the moment unit-driven conversion
    existed: reporting "px" for a value the export already divided into
    millimetres would be worse than reporting nothing.
    """
    assert cf.export_units(calibrated)[LENGTH] == "px"
    assert cf.export_units(calibrated)["traits__organism__area_px"] == "px2"
