"""
A score fitted to human labels.

Clustering then picking clusters by eye is a person fitting a classifier by
hand. With screening labels stored in the project, the same judgement can be
fitted: a model over stored features predicting the label, its probability
stored like any metric and thresholded like any QC score.

What is tested here rather than described:

  the labelled occurrences are scored by a model that did not see their own
  label. They are the rows a threshold gets calibrated on, and scored in-sample
  they would flatter every cutoff;

  which labels train the model has to be said, so the labels a filter is
  audited on cannot be folded in by leaving an argument out;

  the fit is recorded beside the recipe hash, including a digest of which
  occurrence carried which label, since the same occurrences relabelled are a
  different fit.
"""

import pytest

import critterframe as cf
from critterframe.metrics.label_score import MIN_PER_CLASS
from critterframe.metrics.run import RunContext
from critterframe.metrics.stored import StoredValues
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.recipes import Recipe

pytest.importorskip("sklearn")

IDS = [f"specimen{index}" for index in range(8)]

# Short bodies are good, long ones bad: separable on one feature.
LENGTHS = dict(zip(IDS, [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]))
LABELS = dict(zip(IDS, ["good", "good", "good", "good",
                        "wrong_region", "incomplete", "overflow", "input_invalid"]))


# The screening metric the labels are stored under: a vocabulary of the project's own.
QUALITY = cf.exclusive_label_annotation(
    ["good", "input_invalid", "wrong_region", "incomplete", "overflow"], name="quality")


def store(project_path, run_name, operation, values):
    recipe = Recipe("metric", run_name, [operation], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(project_path, run_id, recipe.hash,
                   [make_metric_row(occurrence_id, "organism", operation.metric_name, value)
                    for occurrence_id, value in values.items()])


def labelled(project_path, lengths=LENGTHS, labels=LABELS):
    store(project_path, "traits", cf.body_length(), lengths)
    store(project_path, "quality", QUALITY, labels)
    cf.define_subset(project_path, "calibrate", occurrence_ids=list(labels))


def a_metric(**kwargs):
    kwargs.setdefault("labels_subset", "calibrate")
    kwargs.setdefault("label_metric", "quality")
    if "bad_labels" not in kwargs:
        kwargs.setdefault("good_labels", ["good"])
    return cf.label_score([cf.body_length()], from_run="traits",
                          labels_run="quality", **kwargs)


def a_context(project_path, occurrence_ids=IDS):
    return RunContext(project_path, occurrence_ids, "organism", "bad_score")


def scored(metric, occurrence_id):
    return metric(StoredValues(occurrence_id, "organism"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_which_labels_train_the_score_has_to_be_said():
    """
    No default: the labels left out are what a filter on this score is audited
    against, and a default of "all of them" would use those up silently.
    """
    with pytest.raises(TypeError):
        cf.label_score([cf.body_length()], from_run="traits", labels_run="quality")


def test_what_the_score_is_fitted_to_is_in_the_hash():
    base = a_metric()
    assert base.spec() == a_metric().spec()
    assert base.spec() != a_metric(labels_subset="other").spec()
    assert base.spec() != a_metric(bad_labels=["wrong_region"]).spec()
    assert base.spec() != a_metric(n_components=4).spec()
    assert base.spec() != a_metric(label_metric="other_label").spec()
    assert base.spec() != cf.label_score([cf.max_width()], from_run="traits",
                                         labels_run="quality",
                                         labels_subset="calibrate",
                                         label_metric="quality",
                                         good_labels=["good"]).spec()


def test_which_labels_are_bad_has_to_be_said_one_way():
    """
    Listing the good labels or listing the bad ones, never neither and never
    both: the score predicts one side of a line, and the caller draws it.
    """
    for flags in ({}, {"bad_labels": ["wrong_region"], "good_labels": ["good"]}):
        with pytest.raises(ValueError, match="exactly one of"):
            cf.label_score([cf.body_length()], from_run="traits", labels_run="quality",
                           labels_subset="calibrate", label_metric="quality", **flags)


def test_listing_the_bad_labels_predicts_only_those(metadata_project):
    """Under bad_labels a label left off the list is one the model calls fine."""
    labelled(metadata_project)
    listed = a_metric(bad_labels=["wrong_region", "incomplete", "overflow", "input_invalid"])
    narrower = a_metric(bad_labels=["wrong_region", "incomplete", "overflow"])

    assert listed.prepare(a_context(metadata_project))["n_bad"] == 4
    assert narrower.prepare(a_context(metadata_project))["n_bad"] == 3


def test_scoring_without_fitting_first_says_so():
    with pytest.raises(RuntimeError, match="never fit"):
        scored(a_metric(), "specimen0")


def test_a_label_score_reads_stored_values_not_pixels():
    assert a_metric().input == "stored"


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_bad_segments_score_higher_than_good_ones(metadata_project):
    labelled(metadata_project)
    metric = a_metric()
    metric.prepare(a_context(metadata_project))

    good = [scored(metric, occurrence_id)["bad_probability"] for occurrence_id in IDS[:4]]
    bad = [scored(metric, occurrence_id)["bad_probability"] for occurrence_id in IDS[4:]]
    assert max(good) < 0.5 < min(bad)


def test_a_labelled_occurrence_is_scored_by_a_model_that_did_not_see_it(metadata_project):
    """
    The stored score of a training row is its out-of-fold prediction, not what
    the model fit on every label says about it.
    """
    labelled(metadata_project)
    metric = a_metric()
    metric.prepare(a_context(metadata_project))

    stored = scored(metric, "specimen3")
    in_sample = float(metric.classifier.predict_proba([[LENGTHS["specimen3"]]])[0, 1])

    assert stored["in_training"] is True
    assert stored["bad_probability"] != pytest.approx(in_sample, abs=1e-9)


def test_an_unlabelled_occurrence_is_scored_by_the_full_model(metadata_project):
    labels = {occurrence_id: LABELS[occurrence_id]
              for occurrence_id in IDS if occurrence_id not in ("specimen3", "specimen7")}
    labelled(metadata_project, labels=labels)
    metric = a_metric()
    metric.prepare(a_context(metadata_project))

    for occurrence_id in ("specimen3", "specimen7"):
        stored = scored(metric, occurrence_id)
        expected = float(metric.classifier.predict_proba([[LENGTHS[occurrence_id]]])[0, 1])
        assert stored["in_training"] is False
        assert stored["bad_probability"] == pytest.approx(expected)


def test_only_the_named_subsets_labels_train_it(metadata_project):
    labelled(metadata_project)
    cf.define_subset(metadata_project, "calibrate",
                     occurrence_ids=[occurrence_id for occurrence_id in IDS
                                     if occurrence_id not in ("specimen0", "specimen4")])
    metric = a_metric()
    record = metric.prepare(a_context(metadata_project))

    assert (record["n_good"], record["n_bad"]) == (3, 3)
    assert record["population"]["count"] == 6
    assert scored(metric, "specimen0")["in_training"] is False


def test_the_labelled_occurrences_need_not_be_in_the_run(metadata_project):
    """
    A run over part of the project is still scored by a model fit on all the
    training labels: who is being scored and who trained it are two questions.
    """
    labelled(metadata_project)
    metric = a_metric()
    record = metric.prepare(a_context(metadata_project, ["specimen0", "specimen4"]))

    assert record["population"]["count"] == 8
    assert scored(metric, "specimen4")["bad_probability"] > 0.5


def test_too_few_labels_of_one_kind_is_refused(metadata_project):
    labels = dict(LABELS, specimen5="good", specimen6="good")
    labelled(metadata_project, labels=labels)
    assert sum(label != "good" for label in labels.values()) < MIN_PER_CLASS

    with pytest.raises(ValueError, match="2 bad and 6 good"):
        a_metric().prepare(a_context(metadata_project))


def test_an_occurrence_with_no_feature_value_has_no_score(metadata_project):
    from critterframe.drivers import NoInput

    lengths = {occurrence_id: value for occurrence_id, value in LENGTHS.items()
               if occurrence_id != "specimen0"}
    labelled(metadata_project, lengths=lengths)
    metric = a_metric()
    record = metric.prepare(a_context(metadata_project))

    assert record["population"]["count"] == 7
    with pytest.raises(NoInput):
        scored(metric, "specimen0")


def test_more_components_than_the_labels_allow_is_capped(metadata_project):
    """One feature and eight labels cannot support sixteen components."""
    labelled(metadata_project)
    record = a_metric(n_components=16).prepare(a_context(metadata_project))
    assert record["n_components"] == 1


# ---------------------------------------------------------------------------
# What the run records
# ---------------------------------------------------------------------------


def test_the_fit_record_says_what_it_was_fitted_on(metadata_project):
    labelled(metadata_project)
    record = a_metric().prepare(a_context(metadata_project))

    assert record["from_run"] == "traits"
    assert record["labels_run"] == "quality"
    assert record["labels_subset"] == "calibrate"
    assert (record["n_bad"], record["n_good"]) == (4, 4)
    assert record["cv_auc"] == 1.0
    assert record["from_recipe_hash"]


def test_the_same_occurrences_relabelled_are_a_different_fit(metadata_project):
    """
    Which is what the population digest alone cannot see: it names who was
    labelled, not what they were called.
    """
    labelled(metadata_project)
    before = a_metric().prepare(a_context(metadata_project))

    store(metadata_project, "quality", QUALITY,
          {"specimen3": "incomplete", "specimen4": "good"})
    after = a_metric().prepare(a_context(metadata_project))

    assert after["population"] == before["population"]
    assert after["fit_hash"] != before["fit_hash"]


def test_one_bad_label_for_another_is_the_same_fit(metadata_project):
    """The model predicts bad or not; which kind of bad does not reach it."""
    labelled(metadata_project)
    before = a_metric().prepare(a_context(metadata_project))

    store(metadata_project, "quality", QUALITY,
          {"specimen4": "overflow"})
    after = a_metric().prepare(a_context(metadata_project))

    assert after["fit_hash"] == before["fit_hash"]
