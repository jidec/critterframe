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
    return pd.DataFrame(
        {
            "occurrence_id": [f"occ{index}" for index in range(10)],
            "qc__organism__blur_variance": [5.0, 8.0, 12.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "screening__organism__usability": [
                "not_an_organism",
                "cut_off",
                "cut_off",
                "usable",
                "usable",
                "usable",
                "usable",
                "usable",
                "usable",
                "usable",
            ],
        }
    )


METRIC = "qc__organism__blur_variance"
FLAG = "screening__organism__usability"

# The image-screening vocabulary the first sections use, and how a call names it.
# Nothing in the package knows these labels, so every call says them.
SCREEN_BAD = ("not_an_organism", "cut_off")
SCREEN = {"label_metric": "usability", "bad_labels": SCREEN_BAD}

# The bad labels of the screening vocabulary the later sections use: every label
# but "good". A project's own vocabulary, so the tests name it themselves.
QUALITY_BAD = ("input_invalid", "wrong_region", "incomplete", "overflow")


def sweep():
    return sweep_thresholds(labelled(), METRIC, "below", FLAG, SCREEN_BAD)


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
    assert row["recall_not_an_organism"] == 1.0  # occ0 at 5.0
    assert row["recall_cut_off"] == pytest.approx(0.5)  # occ1 at 8.0, not occ2
    assert row["n_cut_off"] == 2


def test_above_flags_the_other_side():
    """
    Blur variance is lower for worse images; asymmetry and edge fraction are
    higher. Writing the direction out at every call site is how it eventually
    gets written backwards, which is why the shorthand exists.
    """
    rows = sweep_thresholds(labelled(), METRIC, "above", FLAG, SCREEN_BAD)
    row = rows.set_index("threshold").loc[20.0]
    assert row["n_flagged"] == 6  # 30, 40, 50, 60, 70, 80
    assert row["precision"] == 0.0  # all of them usable


def test_an_unmeasured_or_unlabelled_row_is_dropped():
    """You cannot score what wasn't measured or wasn't labelled."""
    frame = labelled()
    frame.loc[0, METRIC] = None
    frame.loc[9, FLAG] = None

    rows = sweep_thresholds(frame, METRIC, "below", FLAG, SCREEN_BAD)
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
        assert sweep_thresholds(frame, METRIC, "below", FLAG, SCREEN_BAD).empty
    assert "nothing to sweep" in caplog.text


def test_a_bad_direction_raises():
    with pytest.raises(ValueError, match='"below" or "above"'):
        sweep_thresholds(labelled(), METRIC, "beneath", FLAG, SCREEN_BAD)


@pytest.mark.parametrize(
    "metric_col, label_col",
    [
        ("nope", FLAG),
        (METRIC, "nope"),
    ],
)
def test_a_missing_column_raises_and_lists_what_is_there(metric_col, label_col):
    with pytest.raises(KeyError, match="not in the metrics frame"):
        sweep_thresholds(labelled(), metric_col, "below", label_col, SCREEN_BAD)


def test_the_bad_labels_are_configurable():
    """
    Which labels count as "should have been filtered" is a project's judgement,
    not this module's.
    """
    rows = sweep_thresholds(labelled(), METRIC, "below", FLAG, bad_labels=["cut_off"])
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
    """ "Don't throw away more than 15% of good data" is max_fpr=0.15."""
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
    assert {"threshold", "precision", "recall", "fpr", "n_bad", "n_clean"} <= set(chosen.index)


# ---------------------------------------------------------------------------
# _resolve_specs
# ---------------------------------------------------------------------------


def test_a_known_metric_carries_its_own_direction():
    """
    "Is a higher edge_fraction worse" is a property of the metric rather than a
    decision a caller makes.
    """
    assert _resolve_specs(["blur_variance", "edge_fraction"]) == {
        "blur_variance": "below",
        "edge_fraction": "above",
    }


def test_an_unknown_metric_has_to_be_told_which_way_round():
    with pytest.raises(KeyError, match="which side means flag it"):
        _resolve_specs(["my_custom_score"])


def test_a_dict_says_it_explicitly():
    assert _resolve_specs({"my_custom_score": "above"}) == {"my_custom_score": "above"}


# ---------------------------------------------------------------------------
# get_validated_filters(visualize=)
# ---------------------------------------------------------------------------


def _store(project_path, run_name, metric_name, values, part="organism"):
    import critterframe as cf
    from critterframe.recipes import Recipe
    from critterframe.records.metrics import append_metrics, make_metric_row
    from critterframe.records.runs import start_run

    recipe = Recipe("metric", run_name, [cf.body_length()], part=part)
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [make_metric_row(occurrence_id, part, metric_name, value) for occurrence_id, value in values.items()],
    )


def _labelled_project(project_path):
    frame = labelled().head(6)
    frame["occurrence_id"] = [f"specimen{index}" for index in range(6)]
    _store(project_path, "qc", "blur_variance", dict(zip(frame["occurrence_id"], frame[METRIC])))
    _store(project_path, "screening", "usability", dict(zip(frame["occurrence_id"], frame[FLAG])))


def test_visualize_writes_one_sweep_figure_per_metric(metadata_project):
    from critterframe.project import paths
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    filters = get_validated_filters(
        metadata_project, ["blur_variance"], "qc", "screening", **SCREEN, max_fpr=0.5
    )

    assert filters
    figures = list(
        paths.pipeline_dir(metadata_project).glob("filters__qc__vs__screening_*__blur_variance.png")
    )
    assert len(figures) == 1


def test_visualize_false_writes_no_figure(metadata_project):
    from critterframe.project import paths
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    get_validated_filters(
        metadata_project, ["blur_variance"], "qc", "screening", **SCREEN, max_fpr=0.5, visualize=False
    )
    assert not paths.pipeline_dir(metadata_project).exists()


# ---------------------------------------------------------------------------
# get_validated_filters(subset=, label_part=)
# ---------------------------------------------------------------------------


def test_a_subset_restricts_which_labels_calibrate(metadata_project):
    """
    What keeps an audit honest: the labels a threshold is chosen on and the
    labels it is scored on have to be different ones, so calibration has to be
    able to leave some alone.
    """
    import critterframe as cf
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    cf.define_subset(
        metadata_project, "calibrate", occurrence_ids=["specimen2", "specimen3", "specimen4", "specimen5"]
    )

    def calibrate(**kwargs):
        return get_validated_filters(
            metadata_project,
            ["blur_variance"],
            "qc",
            "screening",
            **SCREEN,
            max_fpr=0.0,
            visualize=False,
            defaults={"blur_variance": -1.0},
            **kwargs,
        )

    # All six: flagging below 20 catches specimen0-2 and no usable crop.
    assert calibrate() == {METRIC: (">=", 20.0)}
    # The subset holds one bad crop (12.0) and three usable ones, so the same
    # cutoff is still the answer, read off four labels instead of six.
    assert calibrate(subset="calibrate") == {METRIC: (">=", 20.0)}

    cf.define_subset(metadata_project, "clean_only", occurrence_ids=["specimen3", "specimen4", "specimen5"])
    # Nothing bad among these labels, so they justify no cutoff at all and the
    # stated default is what comes back.
    assert calibrate(subset="clean_only") == {METRIC: (">=", -1.0)}


def test_the_subset_is_part_of_a_calibration_reports_identity(metadata_project):
    """Two calibrations on different labels must not overwrite each other's figures."""
    import critterframe as cf
    from critterframe.project import paths
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    cf.define_subset(
        metadata_project, "calibrate", occurrence_ids=["specimen2", "specimen3", "specimen4", "specimen5"]
    )
    for subset in (None, "calibrate"):
        get_validated_filters(
            metadata_project, ["blur_variance"], "qc", "screening", **SCREEN, max_fpr=0.5, subset=subset
        )

    assert len(list(paths.pipeline_dir(metadata_project).glob("*__blur_variance.png"))) == 2


def test_labels_on_one_part_can_calibrate_a_score_on_another(metadata_project):
    """
    A bad organism mask takes every part below it down, so an organism-level
    score is a legitimate candidate for filtering an abdomen.
    """
    from critterframe.validation.filters import get_validated_filters

    ids = [f"specimen{index}" for index in range(6)]
    _store(metadata_project, "qc", "blur_variance", dict(zip(ids, [5.0, 8.0, 12.0, 20.0, 30.0, 40.0])))
    _store(
        metadata_project,
        "quality",
        "quality",
        dict(zip(ids, ["wrong_region", "input_invalid", "overflow", "good", "good", "good"])),
        part="abdomen",
    )

    filters = get_validated_filters(
        metadata_project,
        ["blur_variance"],
        "qc",
        "quality",
        label_metric="quality",
        label_part="abdomen",
        bad_labels=QUALITY_BAD,
        max_fpr=0.0,
        visualize=False,
    )

    assert filters == {METRIC: (">=", 20.0)}


# ---------------------------------------------------------------------------
# audit_filters
# ---------------------------------------------------------------------------

QUALITY = "quality__organism__quality"
SCORE = "qc__organism__score"
OTHER = "qc2__organism__other"


def _audited_project(project_path):
    """
    Eight screened segments. Three are bad, and `score` separates two of them:

      specimen  0             1              2     3           4..7
      label     wrong_region  input_invalid  good  incomplete  good
      score     1             2              3     4           5..8
      other     1             1              --    1           1
    """
    ids = [f"specimen{index}" for index in range(8)]
    labels = ["wrong_region", "input_invalid", "good", "incomplete", "good", "good", "good", "good"]
    _store(project_path, "qc", "score", dict(zip(ids, range(1, 9))))
    _store(
        project_path,
        "qc2",
        "other",
        {occurrence_id: 1 for occurrence_id in ids if occurrence_id != "specimen2"},
    )
    _store(project_path, "quality", "quality", dict(zip(ids, labels)))


def _audit(project_path, filters, **kwargs):
    from critterframe.validation.filters import audit_filters

    kwargs.setdefault("visualize", False)
    kwargs.setdefault("label_metric", "quality")
    if "good_labels" not in kwargs:
        kwargs.setdefault("bad_labels", QUALITY_BAD)
    return audit_filters(project_path, filters, "quality", **kwargs)


def test_an_audit_reports_what_is_left_in_what_was_kept(metadata_project):
    """
    The number a filter set exists to move. score >= 3 removes specimen0 and
    specimen1, both bad, and keeps six rows of which specimen3 is still bad.
    """
    _audited_project(metadata_project)
    result = _audit(metadata_project, {SCORE: (">=", 3)})

    assert (result["n"], result["n_bad"], result["n_kept"], result["n_kept_bad"]) == (8, 3, 6, 1)
    assert result["coverage"] == pytest.approx(6 / 8)
    assert result["bad_rate_before"] == pytest.approx(3 / 8)
    assert result["bad_rate_after"] == pytest.approx(1 / 6)
    assert result["good_retained"] == 1.0
    assert result["bad_caught"] == pytest.approx(2 / 3)
    assert result["kept_bad"] == ["specimen3"]
    assert result["dropped_good"] == []


def test_what_was_caught_is_broken_out_per_reason(metadata_project):
    """
    A filter set that removes every invalid input and no segmenter failure is a
    different instrument from one that removes half of each.
    """
    _audited_project(metadata_project)
    result = _audit(metadata_project, {SCORE: (">=", 3)})

    assert result["recall_wrong_region"] == 1.0
    assert result["recall_input_invalid"] == 1.0
    assert result["recall_incomplete"] == 0.0
    assert result["n_incomplete"] == 1
    assert np.isnan(result["recall_overflow"]) and result["n_overflow"] == 0


def test_every_rate_carries_an_interval(metadata_project):
    """
    One bad row in six kept is 17%, and from six rows that could be anything
    from 3% to 56% -- the difference between a result and an anecdote.
    """
    _audited_project(metadata_project)
    result = _audit(metadata_project, {SCORE: (">=", 3)})

    low, high = result["bad_rate_after_ci"]
    assert low < result["bad_rate_after"] < high
    assert (low, high) == pytest.approx((0.0301, 0.5635), abs=1e-3)
    assert result["good_retained_ci"][1] == 1.0


def test_the_wilson_interval_stays_inside_zero_and_one():
    from critterframe.validation.filters import _wilson

    assert _wilson(0, 10)[0] == 0.0 and 0 < _wilson(0, 10)[1] < 0.5
    assert _wilson(10, 10)[1] == 1.0
    assert all(np.isnan(bound) for bound in _wilson(0, 0))


def test_an_unmeasured_filter_column_counts_as_removed(metadata_project):
    """
    The export's own rule: a missing value never passes. specimen2 is good and
    has no `other`, so a filter on it costs a good row, and the audit says so.
    """
    _audited_project(metadata_project)
    result = _audit(metadata_project, {SCORE: (">=", 3), OTHER: ("==", 1)})

    assert result["dropped_good"] == ["specimen2"]
    assert result["n_kept"] == 5
    assert result["good_retained"] == pytest.approx(4 / 5)


def test_each_filter_is_scored_alone_and_for_what_only_it_removes(metadata_project):
    """
    Two filters removing the same rows are one filter and a redundant one, and
    the combined number cannot show which.
    """
    _audited_project(metadata_project)
    result = _audit(metadata_project, {SCORE: (">=", 3), OTHER: ("==", 1)})
    table = result["filters"].set_index("filter")

    assert table.loc[SCORE, ["removed", "removed_bad", "removed_good"]].tolist() == [2, 2, 0]
    assert table.loc[OTHER, ["removed", "removed_bad", "removed_good"]].tolist() == [1, 0, 1]
    assert table["removed_only_here"].tolist() == [2, 1]

    redundant = _audit(metadata_project, {SCORE: (">=", 3), QUALITY: ("!=", "wrong_region")})
    assert redundant["filters"].set_index("filter").loc[QUALITY, "removed_only_here"] == 0


def below_three(values):
    return values < 3


def test_membership_and_predicate_filters_are_audited_too(metadata_project):
    """Whatever export_metrics accepts, since it is the export being audited."""
    _audited_project(metadata_project)

    kept_out = _audit(metadata_project, {QUALITY: ("not in", ["incomplete"])})
    assert kept_out["n_kept"] == 7

    flipped = _audit(metadata_project, {SCORE: below_three})
    assert flipped["n_kept"] == 2 and flipped["bad_rate_after"] == 1.0


def test_no_filters_is_the_unfiltered_baseline(metadata_project):
    _audited_project(metadata_project)
    result = _audit(metadata_project, {})

    assert result["coverage"] == 1.0
    assert result["bad_rate_after"] == result["bad_rate_before"]
    assert result["filters"].empty


def test_an_audit_reads_only_its_subset(metadata_project):
    import critterframe as cf

    _audited_project(metadata_project)
    cf.define_subset(metadata_project, "audit", occurrence_ids=["specimen3", "specimen4", "specimen5"])
    result = _audit(metadata_project, {SCORE: (">=", 3)}, subset="audit")

    assert (result["n"], result["n_bad"], result["n_kept"]) == (3, 1, 3)


def test_an_audit_can_read_only_the_runs_it_needs(metadata_project):
    """
    On a large project every stored value is a lot to read for a handful of
    columns. Naming the runs gives the same answer, and leaving one out that a
    filter needs is an error rather than a filter silently not applied.
    """
    _audited_project(metadata_project)
    filters = {SCORE: (">=", 3)}

    narrowed = _audit(metadata_project, filters, run_names=["qc", "quality"])
    assert narrowed["n_kept"] == _audit(metadata_project, filters)["n_kept"]

    with pytest.raises(KeyError, match="not in the export"):
        _audit(metadata_project, filters, run_names=["quality"])


def test_an_audit_keeps_exactly_what_the_export_keeps(metadata_project):
    """The property that makes the audit about the export and not beside it."""
    import critterframe as cf

    _audited_project(metadata_project)
    filters = {SCORE: (">=", 3), OTHER: ("==", 1)}
    exported = cf.export_metrics(metadata_project, path=False, manifest=False, filters=filters)

    assert _audit(metadata_project, filters)["n_kept"] == len(exported)


def test_a_filter_on_a_column_that_does_not_exist_raises(metadata_project):
    _audited_project(metadata_project)
    with pytest.raises(KeyError, match="not in the export"):
        _audit(metadata_project, {"qc__organism__nope": (">", 0)})


def test_nothing_labelled_is_an_empty_audit(metadata_project, caplog):
    """Before anyone has screened anything -- an answer, not an error."""
    _store(metadata_project, "qc", "score", {"specimen0": 1})
    with caplog.at_level("WARNING"):
        assert _audit(metadata_project, {SCORE: (">=", 3)}) == {}
    assert "nothing to audit" in caplog.text


def test_a_small_audit_says_it_is_small(metadata_project, caplog):
    _audited_project(metadata_project)
    with caplog.at_level("WARNING"):
        _audit(metadata_project, {SCORE: (">=", 3)})
    assert "too wide to support a claim" in caplog.text


def test_an_audit_persists_nothing(metadata_project):
    """A validation score is a judgement about a method, not a trait."""
    from critterframe.project import paths
    from critterframe.export import load_exports
    from critterframe.records.runs import load_runs

    _audited_project(metadata_project)
    runs_before = len(load_runs(metadata_project))
    _audit(metadata_project, {SCORE: (">=", 3)})

    assert len(load_runs(metadata_project)) == runs_before
    assert len(load_exports(metadata_project)) == 0
    assert not paths.pipeline_dir(metadata_project).exists()


def test_an_audit_draws_the_funnel_and_the_labels(metadata_project):
    from critterframe.project import paths

    _audited_project(metadata_project)
    _audit(metadata_project, {SCORE: (">=", 3)}, visualize=True)

    names = {
        path.name.split("__")[-1]
        for path in paths.pipeline_dir(metadata_project).glob("audit_filters__quality_*.png")
    }
    assert names == {"funnel.png", "labels.png"}


# ---------------------------------------------------------------------------
# get_validated_filters: a candidate that separates nothing
# ---------------------------------------------------------------------------


def test_a_score_that_catches_nothing_is_left_out(metadata_project, caplog):
    """
    Flagging HIGH blur variance catches no bad crop here, since the bad ones are
    the low ones. A cutoff would still come back, at the top of the labelled
    range, and at export it would remove every unlabelled occurrence beyond it
    on no evidence at all.
    """
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    with caplog.at_level("WARNING"):
        filters = get_validated_filters(
            metadata_project,
            {"blur_variance": "above"},
            "qc",
            "screening",
            **SCREEN,
            max_fpr=0.0,
            visualize=False,
        )

    assert filters == {}
    assert "separates nothing" in caplog.text


def test_a_score_that_catches_one_bad_label_is_kept(metadata_project):
    import critterframe as cf
    from critterframe.validation.filters import get_validated_filters

    _labelled_project(metadata_project)
    cf.define_subset(
        metadata_project,
        "one_bad",  # one bad, three usable
        occurrence_ids=["specimen2", "specimen3", "specimen4", "specimen5"],
    )

    assert get_validated_filters(
        metadata_project,
        {"blur_variance": "below"},
        "qc",
        "screening",
        **SCREEN,
        max_fpr=0.0,
        subset="one_bad",
        visualize=False,
    ) == {METRIC: (">=", 20.0)}


# ---------------------------------------------------------------------------
# get_validated_filters: a categorical candidate
# ---------------------------------------------------------------------------

CLUSTER = "clusters__organism__cluster__cluster_id"


def _clustered_project(project_path):
    """
    Eight screened segments in three clusters:

      cluster   0                        1                        2
      specimen  0     1     2            3     4     5     6      7
      label     bad   bad   good         good  good  good  bad    bad
      bad rate  2/3                      1/4                      1/1

    Four bad, four good. Dropping cluster 0 catches half the bad at the cost of
    a quarter of the good; dropping cluster 1 as well costs every good row.
    """
    ids = [f"specimen{index}" for index in range(8)]
    clusters = [0, 0, 0, 1, 1, 1, 1, 2]
    labels = ["wrong_region", "incomplete", "good", "good", "good", "good", "overflow", "input_invalid"]
    _store(
        project_path,
        "clusters",
        "cluster",
        {
            occurrence_id: {"cluster_id": cluster, "group": None}
            for occurrence_id, cluster in zip(ids, clusters)
        },
    )
    _store(project_path, "quality", "quality", dict(zip(ids, labels)))


def _categories(project_path, **kwargs):
    from critterframe.validation.filters import get_validated_filters

    kwargs.setdefault("visualize", False)
    kwargs.setdefault("max_fpr", 0.25)
    kwargs.setdefault("min_labelled", 2)
    return get_validated_filters(
        project_path,
        {"cluster__cluster_id": "category"},
        "clusters",
        "quality",
        label_metric="quality",
        bad_labels=QUALITY_BAD,
        **kwargs,
    )


def test_the_worst_clusters_are_dropped_within_the_same_budget(metadata_project):
    """
    Picking clusters to keep by eye, done by counting, and held to the constraint
    a threshold is held to: cluster 0 costs a quarter of the good rows and goes.
    """
    _clustered_project(metadata_project)
    assert _categories(metadata_project) == {CLUSTER: ("not in", [0])}


def test_a_mostly_good_cluster_is_not_dropped_for_its_bad_quarter(metadata_project):
    """
    Cluster 1 is a quarter bad. Dropping it for that would throw away three
    good rows to catch one, which is the trade the budget exists to refuse.
    """
    _clustered_project(metadata_project)
    assert 1 not in _categories(metadata_project)[CLUSTER][1]
    assert _categories(metadata_project, max_fpr=1.0) == {CLUSTER: ("not in", [0, 1])}


def test_a_tighter_budget_drops_nothing_and_says_so(metadata_project, caplog):
    _clustered_project(metadata_project)
    with caplog.at_level("WARNING"):
        assert _categories(metadata_project, max_fpr=0.1) == {}
    assert "left out" in caplog.text


def test_a_precision_constraint_reads_the_same_sweep(metadata_project):
    """Two of the three rows cluster 0 removes are bad; with cluster 1 it is three of seven."""
    _clustered_project(metadata_project)
    assert _categories(metadata_project, max_fpr=None, min_precision=0.6) == {CLUSTER: ("not in", [0])}


def test_a_cluster_with_too_few_labels_is_never_dropped(metadata_project, caplog):
    """
    Cluster 2 is all bad on the strength of one label. That is not evidence,
    and a verdict invented from it would be worse than saying so.
    """
    _clustered_project(metadata_project)
    with caplog.at_level("WARNING"):
        filters = _categories(metadata_project, max_fpr=1.0)

    assert 2 not in filters[CLUSTER][1]
    assert "label more to decide" in caplog.text

    assert _categories(metadata_project, min_labelled=1) == {CLUSTER: ("not in", [2, 0])}


def test_the_dropped_clusters_are_plain_values_an_export_can_record(metadata_project):
    """
    They go into an export manifest, and a cluster id read back out of a frame
    with missing rows is the float 0.0, which is not what anyone called it.
    """
    _clustered_project(metadata_project)
    dropped = _categories(metadata_project, min_labelled=1)[CLUSTER][1]
    assert all(type(category) is int for category in dropped)


def test_a_category_filter_narrows_the_export_and_keeps_unseen_clusters(metadata_project):
    """
    "not in" rather than "in": a cluster no label landed in is kept, since the
    labels said nothing against it.
    """
    import critterframe as cf

    _clustered_project(metadata_project)
    cf.define_subset(metadata_project, "calibrate", occurrence_ids=[f"specimen{index}" for index in range(7)])
    filters = _categories(metadata_project, subset="calibrate")
    exported = cf.export_metrics(metadata_project, path=False, manifest=False, filters=filters)

    assert filters == {CLUSTER: ("not in", [0])}
    assert sorted(exported["occurrence_id"]) == [f"specimen{index}" for index in range(3, 8)]


def test_a_category_sweep_is_scored_like_a_threshold_sweep():
    """One set of rates for every kind of candidate, so one constraint reads both."""
    from critterframe.validation.filters import _sweep_categories

    frame = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1, 1, 1, 2],
            "label": ["cut_off", "cut_off", "usable", "usable", "usable", "usable", "cut_off", "cut_off"],
        }
    )
    sweep, table = _sweep_categories(frame, "cluster", "label", ["cut_off"], min_labelled=2)

    assert sweep["threshold"].tolist() == [0, 1, 2]
    assert sweep["dropped"].tolist() == [[], [0], [0, 1]]
    assert sweep["recall"].tolist() == [0.0, 0.5, 0.75]
    assert sweep["fpr"].tolist() == [0.0, 0.25, 1.0]
    assert table["category"].tolist() == [2, 0, 1]  # worst first
    assert table["enough_labels"].tolist() == [False, True, True]


def test_clusters_of_one_part_can_be_judged_by_labels_on_another(metadata_project):
    ids = [f"specimen{index}" for index in range(4)]
    _store(
        metadata_project,
        "clusters",
        "cluster",
        {occurrence_id: {"cluster_id": index // 2} for index, occurrence_id in enumerate(ids)},
    )
    _store(
        metadata_project,
        "quality",
        "quality",
        dict(zip(ids, ["overflow", "overflow", "good", "good"])),
        part="abdomen",
    )

    assert _categories(metadata_project, label_part="abdomen") == {CLUSTER: ("not in", [0])}


def test_a_categorical_metric_that_is_not_there_raises(metadata_project):
    from critterframe.validation.filters import get_validated_filters

    _clustered_project(metadata_project)
    with pytest.raises(KeyError, match="not in the metrics frame"):
        get_validated_filters(
            metadata_project,
            {"cluster__nope": "category"},
            "clusters",
            "quality",
            label_metric="quality",
            bad_labels=QUALITY_BAD,
            visualize=False,
        )


def test_a_direction_that_is_not_one_raises(metadata_project):
    from critterframe.validation.filters import get_validated_filters

    with pytest.raises(ValueError, match="must be one of"):
        get_validated_filters(metadata_project, {"blur_variance": "beneath"}, "qc", "screening", **SCREEN)


def test_the_labels_per_category_are_drawn(metadata_project):
    from critterframe.project import paths

    _clustered_project(metadata_project)
    _categories(metadata_project, visualize=True)

    figures = list(
        paths.pipeline_dir(metadata_project).glob("filters__clusters__vs__quality_*__cluster__cluster_id.png")
    )
    assert len(figures) == 1


# ---------------------------------------------------------------------------
# get_validated_filters: several runs, and what the filters do together
# ---------------------------------------------------------------------------

DEPTH = "qcb__organism__depth"


def _two_run_project(project_path):
    """`_audited_project`, plus a second score in its own run that is HIGH for the bad rows."""
    _audited_project(project_path)
    _store(
        project_path,
        "qcb",
        "depth",
        dict(zip([f"specimen{index}" for index in range(8)], [8, 7, 3, 6, 5, 4, 2, 1])),
    )


def _quality_filters(project_path, metric_specs, predicted_run=None, **kwargs):
    from critterframe.validation.filters import get_validated_filters

    kwargs.setdefault("visualize", False)
    kwargs.setdefault("max_fpr", 0.0)
    return get_validated_filters(
        project_path,
        metric_specs,
        predicted_run,
        "quality",
        label_metric="quality",
        bad_labels=QUALITY_BAD,
        **kwargs,
    )


def test_candidates_from_several_runs_calibrate_in_one_call(metadata_project):
    """
    One call rather than one per run, with the same answer: each candidate is
    still held to the constraint alone.
    """
    _two_run_project(metadata_project)

    separately = {
        **_quality_filters(metadata_project, {"score": "below"}, "qc"),
        **_quality_filters(metadata_project, {"depth": "above"}, "qcb"),
    }
    together = _quality_filters(metadata_project, {"qc": {"score": "below"}, "qcb": {"depth": "above"}})

    assert together == separately == {SCORE: (">=", 3.0), DEPTH: ("<=", 5.0)}


def test_specs_keyed_by_run_and_a_named_run_are_one_or_the_other(metadata_project):
    _two_run_project(metadata_project)
    with pytest.raises(ValueError, match="keyed by run"):
        _quality_filters(metadata_project, {"qc": {"score": "below"}}, "qc")
    with pytest.raises(ValueError, match="needs predicted_run"):
        _quality_filters(metadata_project, {"score": "below"})


def test_what_the_filters_do_together_is_logged(metadata_project, caplog):
    """
    Each candidate was held to the budget alone. Only the labels all of them
    were chosen on can say what they cost together, so the call says it.
    """
    _two_run_project(metadata_project)
    with caplog.at_level("INFO"):
        _quality_filters(metadata_project, {"qc": {"score": "below"}, "qcb": {"depth": "above"}})

    # score >= 3 removes specimen0 and specimen1; depth <= 5 removes those two
    # again and specimen3, the bad row score missed. Five good rows are left.
    assert "2 filter(s) together, on the labels they were chosen on (n=8, 3 bad)" in caplog.text
    assert "bad rate 37.5% -> 0.0%" in caplog.text
    assert "removed per reason" in caplog.text
    assert "removed_only_here" in caplog.text


def test_the_labels_before_and_after_are_drawn(metadata_project):
    from critterframe.project import paths

    _two_run_project(metadata_project)
    _quality_filters(metadata_project, {"qc": {"score": "below"}, "qcb": {"depth": "above"}}, visualize=True)

    names = [
        path.name for path in paths.pipeline_dir(metadata_project).glob("filters__qc+qcb__vs__quality_*.png")
    ]
    assert len(names) == 4  # one per candidate, "together", and "tradeoffs"
    assert sum(name.endswith("__together.png") for name in names) == 1
    assert sum(name.endswith("__tradeoffs.png") for name in names) == 1


# ---------------------------------------------------------------------------
# get_validated_filters(audit_subset=)
# ---------------------------------------------------------------------------


def _two_samples(project_path):
    import critterframe as cf

    _audited_project(project_path)
    cf.define_subset(project_path, "calibrate", occurrence_ids=[f"specimen{index}" for index in range(4)])
    cf.define_subset(project_path, "audit", occurrence_ids=[f"specimen{index}" for index in range(4, 8)])


def test_the_chosen_filters_can_be_audited_in_the_same_call(metadata_project, caplog):
    """
    Chosen on one sample, scored on another it never saw: both numbers come out
    of one call, and they are not the same number.
    """
    _two_samples(metadata_project)
    with caplog.at_level("INFO"):
        filters = _quality_filters(
            metadata_project, {"score": "below"}, "qc", subset="calibrate", audit_subset="audit"
        )

    assert filters == {SCORE: (">=", 3.0)}
    assert "on the labels they were chosen on (n=4, 3 bad)" in caplog.text
    assert f"against '{QUALITY}' in subset 'audit' (n=4, 0 bad)" in caplog.text


def test_an_audit_sample_cannot_share_labels_with_calibration(metadata_project):
    import critterframe as cf

    _two_samples(metadata_project)
    cf.define_subset(metadata_project, "overlapping", occurrence_ids=["specimen3", "specimen4"])

    with pytest.raises(ValueError, match="share 1 occurrence"):
        _quality_filters(
            metadata_project, {"score": "below"}, "qc", subset="calibrate", audit_subset="overlapping"
        )
    with pytest.raises(ValueError, match="needs subset="):
        _quality_filters(metadata_project, {"score": "below"}, "qc", audit_subset="audit")


def test_a_category_filter_is_audited_like_any_other(metadata_project):
    from critterframe.validation.filters import audit_filters

    _clustered_project(metadata_project)
    result = audit_filters(
        metadata_project,
        _categories(metadata_project),
        "quality",
        label_metric="quality",
        bad_labels=QUALITY_BAD,
        visualize=False,
    )

    assert (result["n_kept"], result["n_kept_bad"]) == (5, 2)
    assert result["dropped_good"] == ["specimen2"]


# ---------------------------------------------------------------------------
# good_labels: naming the labels that are fine
# ---------------------------------------------------------------------------


def test_naming_the_good_labels_draws_the_same_line(metadata_project):
    """Two ways of saying one thing, where the vocabulary has one good label."""
    _audited_project(metadata_project)
    filters = {SCORE: (">=", 3)}

    by_good = _audit(metadata_project, filters, good_labels=["good"])
    by_bad = _audit(metadata_project, filters)

    assert (by_good["n_bad"], by_good["n_kept_bad"]) == (by_bad["n_bad"], by_bad["n_kept_bad"])
    assert _quality_filters(metadata_project, {"score": "below"}, "qc") == {SCORE: (">=", 3.0)}


def test_a_label_nobody_listed_is_bad_under_good_labels(metadata_project):
    """
    The hazard of listing the bad ones: a label added to the vocabulary and not
    to the list counts as clean, in every sweep and audit, with no error.
    Listing the good ones fails the other way, which is the safe way.
    """
    _audited_project(metadata_project)
    _store(metadata_project, "quality", "quality", {"specimen4": "smudged"})

    assert _audit(metadata_project, {})["n_bad"] == 3
    by_good = _audit(metadata_project, {}, good_labels=["good"])
    assert by_good["n_bad"] == 4
    assert by_good["n_smudged"] == 1


def test_good_and_bad_labels_are_one_or_the_other(metadata_project):
    from critterframe.validation.filters import audit_filters, get_validated_filters

    _audited_project(metadata_project)
    with pytest.raises(ValueError, match="not both"):
        audit_filters(
            metadata_project,
            {},
            "quality",
            label_metric="quality",
            bad_labels=QUALITY_BAD,
            good_labels=["good"],
            visualize=False,
        )
    with pytest.raises(ValueError, match="not both"):
        get_validated_filters(
            metadata_project,
            {"score": "below"},
            "qc",
            "quality",
            label_metric="quality",
            bad_labels=QUALITY_BAD,
            good_labels=["good"],
            visualize=False,
        )


def test_the_labels_and_which_are_bad_have_to_be_said(metadata_project):
    """The package holds no vocabulary, so there is nothing to fall back on."""
    from critterframe.validation.filters import audit_filters, get_validated_filters

    with pytest.raises(ValueError, match="label_metric"):
        get_validated_filters(metadata_project, {"score": "below"}, "qc", "quality", bad_labels=QUALITY_BAD)
    with pytest.raises(ValueError, match="bad_labels= or good_labels="):
        get_validated_filters(metadata_project, {"score": "below"}, "qc", "quality", label_metric="quality")
    with pytest.raises(ValueError, match="label_metric"):
        audit_filters(metadata_project, {}, "quality", bad_labels=QUALITY_BAD)
    with pytest.raises(ValueError, match="bad_labels= or good_labels="):
        audit_filters(metadata_project, {}, "quality", label_metric="quality")


def test_calibrating_with_good_labels_matches_bad_labels(metadata_project):
    from critterframe.validation.filters import get_validated_filters

    _audited_project(metadata_project)
    assert get_validated_filters(
        metadata_project,
        {"score": "below"},
        "qc",
        "quality",
        label_metric="quality",
        good_labels=["good"],
        max_fpr=0.0,
        visualize=False,
    ) == {SCORE: (">=", 3.0)}


# ---------------------------------------------------------------------------
# Redundant filters: what each one removes that nothing else does
# ---------------------------------------------------------------------------


def _overlapping_filters():
    """
    Eight labelled rows and what each of three filters removes (a 1 fails the filter):

      row    0    1    2    3     4     5..7
      label  bad  bad  bad  good  good  good
      x      1    1    .    .     .     .       alone in removing row 0
      y      .    1    1    1     .     .       alone in removing rows 2 and 3
      z      .    .    .    .     1     .       alone in removing row 4, a good one
    """
    frame = pd.DataFrame(
        {
            "occurrence_id": [f"occ{index}" for index in range(8)],
            "label": ["cut_off"] * 3 + ["usable"] * 5,
            "x": [1, 1, 0, 0, 0, 0, 0, 0],
            "y": [0, 1, 1, 1, 0, 0, 0, 0],
            "z": [0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    return frame, {"x": ("==", 0), "y": ("==", 0), "z": ("==", 0)}


def test_what_only_one_filter_removes_is_split_into_bad_and_good():
    """
    The split is what says whether a filter earns its place: unique bad rows are
    what it adds, unique good rows are what it costs.
    """
    from critterframe.validation.filters import _score_filters

    frame, filters = _overlapping_filters()
    table = _score_filters(frame, filters, "label", ["cut_off"])[0]["filters"].set_index("filter")

    assert table.loc["x", ["only_here_bad", "only_here_good"]].tolist() == [1, 0]
    assert table.loc["y", ["only_here_bad", "only_here_good"]].tolist() == [1, 1]
    assert table.loc["z", ["only_here_bad", "only_here_good"]].tolist() == [0, 1]
    assert (table["only_here_bad"] + table["only_here_good"]).tolist() == table["removed_only_here"].tolist()


def test_a_filter_that_only_costs_good_rows_is_dropped():
    """
    z catches no bad row the others miss, so removing it loses nothing and gives
    a good row back. x and y each hold a bad row alone and stay.
    """
    from critterframe.validation.filters import _drop_redundant, _score_filters

    frame, filters = _overlapping_filters()
    before = _score_filters(frame, filters, "label", ["cut_off"])[0]
    pruned = _drop_redundant(frame, filters, "label", ["cut_off"])
    after = _score_filters(frame, pruned, "label", ["cut_off"])[0]

    assert sorted(pruned) == ["x", "y"]
    assert after["bad_caught"] == before["bad_caught"] == 1.0
    assert after["good_retained"] > before["good_retained"]


def test_of_two_filters_that_duplicate_each_other_one_survives():
    """
    Each makes the other redundant, so both look droppable at first. Re-scoring
    after every drop is what keeps the second, and the name breaks the tie so
    the survivor is the same one every time.
    """
    from critterframe.validation.filters import _drop_redundant

    frame, filters = _overlapping_filters()
    frame["x_again"] = frame["x"]
    filters = {"x_again": ("==", 0), **filters}

    for _attempt in range(3):
        assert sorted(_drop_redundant(frame, filters, "label", ["cut_off"])) == ["x_again", "y"]


def test_one_filter_alone_is_never_redundant():
    from critterframe.validation.filters import _drop_redundant

    frame, _filters = _overlapping_filters()
    assert _drop_redundant(frame, {"z": ("==", 0)}, "label", ["cut_off"]) == {"z": ("==", 0)}


def test_redundant_calibrated_filters_are_dropped_only_when_asked(metadata_project, caplog):
    """
    score and depth are calibrated alone and both pass, but depth catches every
    bad row score does and one more. Off by default, since which of two
    correlated scores to keep is a choice someone may want to make themselves.
    """
    _two_run_project(metadata_project)
    specs = {"qc": {"score": "below"}, "qcb": {"depth": "above"}}

    assert sorted(_quality_filters(metadata_project, specs)) == sorted([SCORE, DEPTH])
    with caplog.at_level("INFO"):
        pruned = _quality_filters(metadata_project, specs, drop_redundant=True)

    assert pruned == {DEPTH: ("<=", 5.0)}
    assert f"{SCORE}: dropped as redundant" in caplog.text
    assert "1 filter(s) together" in caplog.text


def test_a_total_budget_warns_and_changes_nothing(metadata_project, caplog):
    """
    Each candidate has its own budget, so the set can cost more than any one of
    them. Flagging score below 5 catches all three bad rows and one good one in
    five: 20% of the good rows.
    """
    _audited_project(metadata_project)

    def calibrate(**kwargs):
        return _quality_filters(metadata_project, {"score": "below"}, "qc", max_fpr=0.5, **kwargs)

    with caplog.at_level("WARNING"):
        within = calibrate(max_total_fpr=0.3)
    assert "over max_total_fpr" not in caplog.text

    with caplog.at_level("WARNING"):
        over = calibrate(max_total_fpr=0.1)
    assert "remove 20.0% of good rows, over max_total_fpr=10.0%" in caplog.text

    assert within == over == calibrate() == {SCORE: (">=", 5.0)}


# ---------------------------------------------------------------------------
# What another constraint would have given
# ---------------------------------------------------------------------------


def test_choosing_a_rule_is_a_lookup_in_a_sweep_already_made():
    """
    The sweep is the expensive part and it does not depend on the constraint,
    so one sweep answers for every setting. The same rules, and the same
    refusals, as the full call.
    """
    from critterframe.validation.filters import _choose_rule

    below = sweep()
    assert _choose_rule(below, "below", "blur_variance", 0.15, None, None)[:2] == ((">=", 20.0), "chosen")
    assert _choose_rule(below, "below", "blur_variance", None, 1.0, None)[:2] == ((">=", 20.0), "chosen")

    # Flagging HIGH blur catches no bad crop: no rule, and no fallback either.
    above = sweep_thresholds(labelled(), METRIC, "above", FLAG, SCREEN_BAD)
    assert _choose_rule(above, "above", "blur_variance", 0.0, None, None)[:2] == (None, "zero recall")

    # Nothing satisfies the constraint: the uncalibrated default where the
    # metric has one, and nothing where it has none.
    rule, why, _choice = _choose_rule(below, "below", "blur_variance", -1, None, None)
    assert why == "fallback" and rule[0] == ">="
    assert _choose_rule(below, "below", "my_score", -1, None, None)[:2] == (None, "no cutoff")
    assert _choose_rule(below, "below", "my_score", -1, None, {"my_score": 7})[:2] == (
        (">=", 7.0),
        "fallback",
    )


def test_a_category_rule_is_read_off_its_sweep_the_same_way():
    from critterframe.validation.filters import _choose_rule, _sweep_categories

    frame = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1, 1, 1, 2],
            "label": ["cut_off", "cut_off", "usable", "usable", "usable", "usable", "cut_off", "cut_off"],
        }
    )
    categories, _table = _sweep_categories(frame, "cluster", "label", ["cut_off"], min_labelled=2)

    assert _choose_rule(categories, "category", None, 0.25, None, None)[:2] == (("not in", [0]), "chosen")
    assert _choose_rule(categories, "category", None, 0.1, None, None)[:2] == (None, "zero recall")
    assert _choose_rule(categories, "category", None, None, 0.99, None)[:2] == (None, "no cutoff")


def _blur_tradeoffs(own=(0.15, None), drop_redundant=False):
    from critterframe.validation.filters import _tradeoff_filters, _tradeoff_table

    frame = labelled()
    swept = [(METRIC, "blur_variance", "below", sweep())]
    combinations = _tradeoff_filters(swept, frame, FLAG, BAD, None, drop_redundant, own)
    return combinations, _tradeoff_table(combinations, frame, FLAG, BAD)


BAD = ["not_an_organism", "cut_off"]


def test_the_grid_holds_this_calls_own_setting_and_not_the_unconstrained_one():
    """
    This call's setting is one row, with the filters the call itself returns.
    No constraint at all is left out: unconstrained, every sweep is won by
    removing everything, which is not a setting anyone is choosing between.
    """
    combinations, table = _blur_tradeoffs(own=(0.15, None))
    by_setting = {(fpr, precision): filters for fpr, precision, filters in combinations}

    assert by_setting[(0.15, None)] == {METRIC: (">=", 20.0)}
    assert (None, None) not in by_setting
    assert "fpr<=0.15" in set(table["setting"])
    assert set(table["setting"]) >= {"fpr<=0.01", "prec>=0.9", "fpr<=0.05 prec>=0.6"}


def test_a_looser_budget_never_catches_less():
    """With precision left alone, more room for cost can only add bad rows caught."""
    _combinations, table = _blur_tradeoffs()
    by_budget = table[table["min_precision"].isna()].sort_values("max_fpr")
    assert by_budget["bad_caught"].is_monotonic_increasing
    assert by_budget["good_retained"].is_monotonic_decreasing


def test_the_two_constraints_are_not_independent():
    """
    Both restrict the same cutoffs and the stricter one decides. Loosening the
    cost budget does nothing while precision is what is holding the cutoff back.
    """
    _combinations, table = _blur_tradeoffs()
    strict = table[table["min_precision"] == 0.9].set_index("max_fpr")["bad_caught"]
    assert strict.loc[0.05] == strict.loc[0.30]

    loose = table[table["min_precision"].isna()].set_index("max_fpr")["bad_caught"]
    assert loose.loc[0.30] >= strict.loc[0.30]


def test_the_frontier_is_the_outcomes_nothing_else_beats():
    from critterframe.validation.filters import _frontier

    table = pd.DataFrame(
        {
            "setting": ["a", "b", "c", "d", "e"],
            "good_retained": [1.0, 0.9, 0.9, 0.8, 0.9],
            "bad_caught": [0.2, 0.6, 0.4, 0.9, 0.6],
        }
    )
    outcomes = _frontier(table).set_index("setting")

    # c keeps as many good rows as b and catches fewer bad ones; e is b again.
    assert outcomes["frontier"].to_dict() == {"a": True, "b": True, "c": False, "d": True}
    assert outcomes.loc["b", "n_same"] == 1
    assert "e" not in outcomes.index


def test_a_calibration_reports_what_other_settings_would_give(metadata_project, caplog):
    from critterframe.project import paths

    _two_run_project(metadata_project)
    with caplog.at_level("INFO"):
        _quality_filters(
            metadata_project, {"qc": {"score": "below"}, "qcb": {"depth": "above"}}, visualize=True
        )

    assert "what other constraints would give, on the calibration labels" in caplog.text
    assert len(list(paths.pipeline_dir(metadata_project).glob("filters__*__tradeoffs.png"))) == 1


def test_the_same_settings_are_scored_on_held_out_labels_when_there_are_some(metadata_project):
    """
    Read off the calibration labels, the best-looking setting is one more thing
    fitted to them. The held-out version is the same grid on labels it never saw.
    """
    from critterframe.project import paths

    import critterframe as cf

    # Each sample needs a bad label: with none, a share of bad rows removed is undefined.
    _audited_project(metadata_project)
    cf.define_subset(
        metadata_project, "calibrate", occurrence_ids=["specimen0", "specimen1", "specimen2", "specimen4"]
    )
    cf.define_subset(
        metadata_project, "audit", occurrence_ids=["specimen3", "specimen5", "specimen6", "specimen7"]
    )
    _quality_filters(
        metadata_project, {"score": "below"}, "qc", subset="calibrate", audit_subset="audit", visualize=True
    )

    written = [path.name for path in paths.pipeline_dir(metadata_project).glob("filters__*.png")]
    assert sum(name.endswith("__tradeoffs.png") for name in written) == 1
    assert sum(name.endswith("__tradeoffs_audit.png") for name in written) == 1
