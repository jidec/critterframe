"""
Comparing two runs' numbers for the same occurrences.

The interesting decisions here are all about what a single agreement figure
hides, and each one has its own column:

  bias vs mean_abs_diff -- a metric 5% high on every specimen is CORRECTABLE; a
  metric randomly 5% off in both directions is not, and one number cannot say
  which you have;

  median vs mean percent difference -- a mean of ratios is unbounded as the
  denominator approaches zero, so one human who clicked twice in the same place
  can move a 37-occurrence mean from 12% to 62%. Both are reported, and neither
  is dropped, because "one reference value is wrong" is a finding rather than
  noise.

Pairing across NAMES rather than making both sides use one name is deliberate
too: storing a human's measurement under the trait's own name would collide with
the trait in the export and misreport where the number came from.
"""

import pytest

import critterframe as cf
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.recipes import Recipe
from critterframe.validation.metrics import _pair_metrics


def store(project_path, run_name, values, metric_name="body_length",
          unit="px"):
    recipe = Recipe("metric", run_name, [cf.body_length()], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(project_path, run_id, recipe.hash,
                   [make_metric_row(occurrence_id, "organism", metric_name,
                                    value, unit=unit)
                    for occurrence_id, value in values.items()])


def ids(count=6):
    return [f"specimen{index}" for index in range(count)]


# ---------------------------------------------------------------------------
# _pair_metrics
# ---------------------------------------------------------------------------


def test_a_list_pairs_each_name_with_itself():
    """The shape of a recipe run twice -- once automatically, once by hand."""
    assert _pair_metrics("auto", "manual", "organism", ["body_length"], []) == [
        ("body_length", "body_length")]


def test_a_mapping_pairs_two_names_for_one_quantity():
    assert _pair_metrics("auto", "manual", "organism",
                         {"body_length": "click_two_points__length_px"}, []) == [
        ("body_length", "click_two_points__length_px")]


def test_no_names_given_compares_everything_the_two_runs_share():
    columns = ["auto__organism__body_length", "auto__organism__max_width",
               "manual__organism__body_length", "manual__organism__mask_area"]
    assert _pair_metrics("auto", "manual", "organism", None, columns) == [
        ("body_length", "body_length")]


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------


def test_identical_values_disagree_about_nothing(metadata_project):
    values = {occurrence_id: 100.0 for occurrence_id in ids()}
    store(metadata_project, "auto", values)
    store(metadata_project, "manual", values)

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["n"] == 6
    assert row["mean_abs_diff"] == 0.0
    assert row["mean_pct_diff"] == 0.0
    assert row["bias"] == 0.0


def test_a_consistent_overestimate_shows_up_as_bias(metadata_project):
    """
    Correctable, and the signed number is what says so -- mean_abs_diff would
    report the same value for a metric that was randomly off in both
    directions.
    """
    reference = {occurrence_id: 100.0 for occurrence_id in ids()}
    predicted = {occurrence_id: 110.0 for occurrence_id in ids()}
    store(metadata_project, "auto", predicted)
    store(metadata_project, "manual", reference)

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["bias"] == pytest.approx(10.0)
    assert row["mean_abs_diff"] == pytest.approx(10.0)
    assert row["mean_pct_diff"] == pytest.approx(10.0)


def test_random_error_cancels_in_the_bias_but_not_the_magnitude(metadata_project):
    store(metadata_project, "auto", dict(zip(ids(2), [110.0, 90.0])))
    store(metadata_project, "manual", dict(zip(ids(2), [100.0, 100.0])))

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["bias"] == pytest.approx(0.0)
    assert row["mean_abs_diff"] == pytest.approx(10.0)


def test_percent_difference_is_relative_to_the_reference_side(metadata_project):
    """
    Which side is the reference is a choice once the names differ, and the
    mapping reads predicted-to-reference -- the same order as the two run
    arguments.
    """
    store(metadata_project, "auto", {"specimen0": 150.0})
    store(metadata_project, "manual", {"specimen0": 100.0})

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["mean_pct_diff"] == pytest.approx(50.0)


def test_the_median_survives_one_impossible_reference(metadata_project, caplog):
    """
    The case the median column exists for: five occurrences agreeing to 1% and
    one whose reference is nearly zero. The mean is dragged into the hundreds;
    the median says the population agrees.
    """
    predicted = {occurrence_id: 100.0 for occurrence_id in ids(6)}
    reference = {occurrence_id: 99.0 for occurrence_id in ids(6)}
    reference["specimen5"] = 0.5              # somebody clicked twice in one spot
    store(metadata_project, "auto", predicted)
    store(metadata_project, "manual", reference)

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["median_pct_diff"] < 5
    assert row["mean_pct_diff"] > row["median_pct_diff"] * 5


def test_an_occurrence_measured_on_only_one_side_is_not_compared(metadata_project):
    store(metadata_project, "auto", {occurrence_id: 100.0
                                     for occurrence_id in ids(6)})
    store(metadata_project, "manual", {occurrence_id: 100.0
                                       for occurrence_id in ids(3)})

    assert cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]["n"] == 3


def test_a_human_measurement_is_compared_across_names(metadata_project):
    """
    The reason mapping exists: the click distance and the measured body length
    are the same quantity recorded two ways, and both keep their own column in
    the export.
    """
    store(metadata_project, "auto", {"specimen0": 100.0})
    store(metadata_project, "manual", {"specimen0": {"length_px": 105.0}},
          metric_name="click_two_points")

    comparison = cf.compare_metrics(
        metadata_project, "auto", "manual",
        metric_names={"body_length": "click_two_points__length_px"})

    row = comparison.iloc[0]
    assert row["metric"] == "body_length"
    assert row["reference_metric"] == "click_two_points__length_px"
    assert row["mean_abs_diff"] == pytest.approx(5.0)


def test_correlation_separates_a_scale_error_from_a_bad_measurement(
        metadata_project):
    """
    High correlation with a large bias means the measurement is fine and the
    scale is off -- which is a completely different problem to fix.
    """
    reference = dict(zip(ids(5), [100.0, 110.0, 120.0, 130.0, 140.0]))
    predicted = {key: value * 2 for key, value in reference.items()}
    store(metadata_project, "auto", predicted)
    store(metadata_project, "manual", reference)

    row = cf.compare_metrics(metadata_project, "auto", "manual").iloc[0]
    assert row["correlation"] == pytest.approx(1.0)
    assert row["bias"] > 100


def test_two_runs_with_nothing_in_common_compare_to_nothing(metadata_project,
                                                            caplog):
    with caplog.at_level("WARNING"):
        assert cf.compare_metrics(metadata_project, "auto", "manual").empty
    assert "no values to compare" in caplog.text


@pytest.mark.slow
def test_stale_predictions_are_left_out_by_default(measured_project):
    """
    Agreement between a current reference value and a predicted value left over
    from an earlier segmentation measures nothing anyone wants a number for --
    it reports the old segmenter's error under the new one's name.
    """
    from helpers.models import ThresholdModel

    cf.run_metrics(measured_project, run_name="manual",
                   metrics=[cf.body_length()], visualize=False)
    assert not cf.compare_metrics(measured_project, "traits", "manual").empty

    cf.run_segments(measured_project, steps=[cf.segment(ThresholdModel(erode=2))],
                    visualize=False)
    assert cf.compare_metrics(measured_project, "traits", "manual").empty
    assert not cf.compare_metrics(measured_project, "traits", "manual",
                                  current_only=False).empty
