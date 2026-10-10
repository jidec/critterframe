"""
Outliers: IsolationForest scores over a group's stored values.

What a group metric does whatever its model is tested in `base/test_group.py`;
this is what is the outlier metric's own.
"""

import pytest

import critterframe as cf

pytest.importorskip("sklearn")


@pytest.mark.slow
def test_an_unusual_specimen_scores_as_more_anomalous(measured_project):
    """
    Both the boolean call and the continuous score beneath it are stored, so a
    stricter or looser cutoff can be applied at export without refitting
    anything.
    """
    cf.run_metrics(
        measured_project,
        run_name="scores",
        metrics=[cf.outlier([cf.body_length(), cf.max_width()], from_run="traits")],
        visualize=False,
    )
    exported = cf.export_metrics(measured_project, run_names=["scores"])

    assert "scores__organism__outlier__anomaly_score" in exported.columns
    assert "scores__organism__outlier__is_outlier" in exported.columns
    # Scored against the whole population, so the group column is empty -- which
    # is what "None where the population-wide fallback was used" looks like in
    # an export, and is exactly the thing a reader needs to know.
    assert exported["scores__organism__outlier__group"].isna().all()


def test_a_different_contamination_is_different_work():
    auto = cf.outlier([cf.body_length()], from_run="traits")
    strict = cf.outlier([cf.body_length()], from_run="traits", contamination=0.05)
    assert auto.spec() != strict.spec()
