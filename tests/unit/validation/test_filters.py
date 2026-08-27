"""
Calibrating a threshold against human labels.

This is the most assertable non-trivial arithmetic in the package: a hand-built
frame with known labels gives exact precision, recall, and false-positive rates,
so every number below is checked against one computed by hand rather than
against whatever the code happened to produce.

What the module is FOR is worth stating, because it explains the shape of the
output. A QC threshold is a judgement about degree, and the only honest way to
pick one is to look at what it would have excluded on data a person has already
labelled. So a sweep reports the cost side (fpr -- real data thrown away) beside
the benefit side (recall), and breaks recall out per label category, because a
metric that catches every cut-off organism while missing every non-organism
would otherwise hide behind one aggregate number.
"""

import numpy as np
import pandas as pd
import pytest

from critterframe.validation.filters import (
    _resolve_specs,
    suggest_threshold,
    sweep_thresholds,
)


def labelled():
    """
    Ten occurrences. Blur variance is LOWER for worse images, so the four bad
    ones sit at the bottom of the range -- separable, but not perfectly.
    """
    return pd.DataFrame({
        "occurrence_id": [f"occ{index}" for index in range(10)],
        "qc__organism__blur_variance": [5.0, 8.0, 12.0, 20.0, 30.0,
                                        40.0, 50.0, 60.0, 70.0, 80.0],
        "screening__organism__annotate_flags": [
            "not_an_organism", "cut_off", "cut_off", "usable", "usable",
            "usable", "usable", "usable", "usable", "usable"],
    })


METRIC = "qc__organism__blur_variance"
FLAG = "screening__organism__annotate_flags"


def sweep():
    return sweep_thresholds(labelled(), METRIC, "below", FLAG)


# ---------------------------------------------------------------------------
# sweep_thresholds
# ---------------------------------------------------------------------------


def test_every_observed_value_is_a_candidate_threshold():
    """
    Observed values rather than a grid: a threshold between two adjacent
    measurements behaves identically to one at the lower of them, and a grid
    would report thresholds no specimen can distinguish.
    """
    rows = sweep()
    assert rows["threshold"].tolist() == sorted(labelled()[METRIC])


def test_the_counts_behind_every_rate_are_reported():
    """
    So a precision computed from two examples is not mistaken for one computed
    from fifty.
    """
    rows = sweep()
    assert set(rows["n_bad"]) == {3}
    assert set(rows["n_clean"]) == {7}


def test_precision_recall_and_fpr_are_what_they_say():
    """
    Checked by hand at a threshold of 20: values BELOW 20 are flagged, which is
    occ0 (5), occ1 (8), occ2 (12). All three are bad, so precision is 1.0,
    recall is 3/3, and no clean occurrence was touched.
    """
    row = sweep().set_index("threshold").loc[20.0]
    assert row["n_flagged"] == 3
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["fpr"] == 0.0


def test_a_looser_threshold_buys_recall_with_false_positives():
    """
    The tradeoff the whole module exists to make visible. At 40, the flagged
    set gains occ3 (20) and occ4 (30), both usable.
    """
    row = sweep().set_index("threshold").loc[40.0]
    assert row["n_flagged"] == 5
    assert row["precision"] == pytest.approx(3 / 5)
    assert row["fpr"] == pytest.approx(2 / 7)


def test_the_tightest_threshold_flags_nothing():
    """The lowest observed value is not below itself."""
    row = sweep().iloc[0]
    assert row["n_flagged"] == 0
    assert np.isnan(row["precision"])
    assert row["recall"] == 0.0


def test_recall_is_broken_out_per_label():
    """
    A metric that catches every cut-off organism while missing every
    non-organism is a different thing from one that catches two thirds of each,
    and the aggregate cannot tell them apart.
    """
    row = sweep().set_index("threshold").loc[12.0]
    assert row["recall_not_an_organism"] == 1.0        # occ0 at 5.0
    assert row["recall_cut_off"] == pytest.approx(0.5)  # occ1 at 8.0, not occ2
    assert row["n_cut_off"] == 2


def test_above_flags_the_other_side():
    """
    Blur variance is lower for worse images; asymmetry and edge fraction are
    higher. Writing the direction out at every call site is how it eventually
    gets written backwards, which is why the shorthand exists.
    """
    rows = sweep_thresholds(labelled(), METRIC, "above", FLAG)
    row = rows.set_index("threshold").loc[20.0]
    assert row["n_flagged"] == 6                       # 30, 40, 50, 60, 70, 80
    assert row["precision"] == 0.0                     # all of them usable


def test_an_unmeasured_or_unlabelled_row_is_dropped():
    """You cannot score what wasn't measured or wasn't labelled."""
    frame = labelled()
    frame.loc[0, METRIC] = None
    frame.loc[9, FLAG] = None

    rows = sweep_thresholds(frame, METRIC, "below", FLAG)
    assert set(rows["n_bad"]) == {2}
    assert set(rows["n_clean"]) == {6}


def test_nothing_labelled_yet_is_an_empty_sweep(caplog):
    """
    The normal state of a project before anyone has done a screening pass --
    an answer, not an error.
    """
    frame = labelled()
    frame[FLAG] = None
    with caplog.at_level("WARNING"):
        assert sweep_thresholds(frame, METRIC, "below", FLAG).empty
    assert "nothing to sweep" in caplog.text


def test_a_bad_direction_raises():
    with pytest.raises(ValueError, match='"below" or "above"'):
        sweep_thresholds(labelled(), METRIC, "beneath", FLAG)


@pytest.mark.parametrize("metric_col, flag_col", [
    ("nope", FLAG),
    (METRIC, "nope"),
])
def test_a_missing_column_raises_and_lists_what_is_there(metric_col, flag_col):
    with pytest.raises(KeyError, match="not in the metrics frame"):
        sweep_thresholds(labelled(), metric_col, "below", flag_col)


def test_the_bad_labels_are_configurable():
    """
    Which labels count as "should have been filtered" is a project's judgement,
    not this module's.
    """
    rows = sweep_thresholds(labelled(), METRIC, "below", FLAG,
                            bad_flags=["cut_off"])
    assert set(rows["n_bad"]) == {2}


# ---------------------------------------------------------------------------
# suggest_threshold
# ---------------------------------------------------------------------------


def test_a_precision_constraint_maximizes_recall_within_it():
    """
    Within a fixed budget for the cost you named, catching more bad data is
    free -- so recall is what is maximized.
    """
    chosen = suggest_threshold(sweep(), min_precision=1.0)
    assert chosen["precision"] == 1.0
    assert chosen["recall"] == 1.0
    assert chosen["threshold"] == 20.0


def test_a_false_positive_budget_is_the_other_way_to_ask():
    """"Don't throw away more than 15% of good data" is max_fpr=0.15."""
    chosen = suggest_threshold(sweep(), max_fpr=0.15)
    assert chosen["fpr"] <= 0.15
    assert chosen["recall"] == 1.0


def test_both_constraints_apply_together():
    chosen = suggest_threshold(sweep(), min_precision=0.9, max_fpr=0.0)
    assert chosen["precision"] >= 0.9 and chosen["fpr"] == 0.0


def test_an_unsatisfiable_constraint_is_itself_the_answer(caplog):
    """
    None means this metric cannot separate the two groups well enough for what
    you asked -- which is a finding, not a failure.
    """
    with caplog.at_level("INFO"):
        assert suggest_threshold(sweep(), min_precision=1.0, max_fpr=-1) is None
    assert "no threshold satisfies" in caplog.text


def test_suggesting_from_an_empty_sweep_is_none():
    assert suggest_threshold(pd.DataFrame()) is None


def test_a_suggestion_is_a_row_you_can_read_the_rest_of():
    """
    A starting point to read off a sweep table, not a recommendation -- so it
    hands back the whole row, counts included.
    """
    chosen = suggest_threshold(sweep(), min_precision=1.0)
    assert {"threshold", "precision", "recall", "fpr", "n_bad",
            "n_clean"} <= set(chosen.index)


# ---------------------------------------------------------------------------
# _resolve_specs
# ---------------------------------------------------------------------------


def test_a_known_metric_carries_its_own_direction():
    """
    "Is a higher edge_fraction worse" is a property of the metric rather than a
    decision a caller makes.
    """
    assert _resolve_specs(["blur_variance", "edge_fraction"]) == {
        "blur_variance": "below", "edge_fraction": "above"}


def test_an_unknown_metric_has_to_be_told_which_way_round():
    with pytest.raises(KeyError, match="which side means flag it"):
        _resolve_specs(["my_custom_score"])


def test_a_dict_says_it_explicitly():
    assert _resolve_specs({"my_custom_score": "above"}) == {
        "my_custom_score": "above"}
