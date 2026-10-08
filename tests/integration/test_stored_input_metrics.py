"""
Metrics whose input is stored values: run_metrics never builds them a segment.

Derived metrics (one occurrence's values) and group metrics (a population's)
both read the metrics table rather than pixels. A recipe of them alone opens no
image store and has nothing for a transform to act on; their currency follows
the upstream run's recipe, recorded beside their own hash.
"""

import pytest

import critterframe as cf
from critterframe.metrics import run as metric_run

SPECIMENS = 8


def width_ratio(values):
    return values["max_width"] / values["body_length"]


def traits(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(
        project_path, run_name="traits", metrics=[cf.body_length(), cf.max_width()], **kwargs
    )["organism"]


def shape(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    metric = cf.derived(width_ratio, [cf.body_length(), cf.max_width()], from_run="traits")
    return cf.run_metrics(project_path, metrics=[metric], **kwargs)["organism"]


def test_a_derived_value_stores_and_exports_like_any_metric(segmented_project):
    traits(segmented_project)
    assert shape(segmented_project)["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False)
    expected = exported["traits__organism__max_width"] / exported["traits__organism__body_length"]
    assert exported["width_ratio__organism__width_ratio"].tolist() == pytest.approx(expected.tolist())


def test_a_stored_only_recipe_opens_no_image_store(segmented_project, monkeypatch):
    traits(segmented_project)

    def refuse(*args, **kwargs):
        raise AssertionError("a stored-input recipe must not read images")

    monkeypatch.setattr(metric_run, "ImageStore", refuse)
    assert shape(segmented_project)["processed"] == SPECIMENS


def test_transforms_have_nothing_to_act_on(segmented_project):
    traits(segmented_project)
    with pytest.raises(ValueError, match="nothing to act on"):
        shape(segmented_project, transforms=[cf.crop_to_mask()])


def test_a_derived_run_is_repeat_aware(segmented_project):
    traits(segmented_project)
    shape(segmented_project)
    again = shape(segmented_project)
    assert (again["processed"], again["skipped"]) == (0, SPECIMENS)


def test_moving_the_upstream_run_rescores_derived_values(segmented_project):
    """Its own hash names only the features; the upstream recipe sits beside it."""
    traits(segmented_project)
    shape(segmented_project)

    traits(segmented_project, transforms=[cf.crop_to_mask()], force=True)
    rescored = shape(segmented_project)

    assert (rescored["processed"], rescored["skipped"]) == (SPECIMENS, 0)


def screen(project_path, monkeypatch, keys, **kwargs):
    """A scripted screening pass: one quality label per occurrence, in id order."""
    from critterframe.metrics import annotation
    from helpers.stubs import FakeCv2

    monkeypatch.setattr(annotation, "cv2", FakeCv2(keys=[ord(key) for key in keys]))
    return cf.run_metrics(
        project_path,
        metrics=[cf.exclusive_label_annotation(["good", "unsure", "bad"], name="quality")],
        visualize=False,
        **kwargs,
    )["organism"]


def bad_score(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    metric = cf.label_score(
        [cf.body_length(), cf.max_width()],
        from_run="traits",
        labels_run="quality",
        labels_subset=None,
        label_metric="quality",
        good_labels=["good"],
    )
    return cf.run_metrics(project_path, metrics=[metric], **kwargs)["organism"]


def test_a_label_score_is_fitted_and_stored_without_reading_an_image(segmented_project, monkeypatch):
    pytest.importorskip("sklearn")
    traits(segmented_project)
    screen(segmented_project, monkeypatch, "11113333")

    def refuse(*args, **kwargs):
        raise AssertionError("a stored-input recipe must not read images")

    monkeypatch.setattr(metric_run, "ImageStore", refuse)
    assert bad_score(segmented_project)["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False, manifest=False)
    assert exported["label_score__organism__label_score__bad_probability"].between(0, 1).all()
    assert exported["label_score__organism__label_score__in_training"].all()

    again = bad_score(segmented_project)
    assert (again["processed"], again["skipped"]) == (0, SPECIMENS)


def test_relabelling_the_same_segments_rescores_them(segmented_project, monkeypatch):
    """
    The same occurrences called something else are a different fit, and a score
    left standing from the old labels would be current in name only. Nothing
    but the labels moved: not the recipe, not the masks, not who was labelled.
    """
    pytest.importorskip("sklearn")
    traits(segmented_project)
    screen(segmented_project, monkeypatch, "11113333")
    bad_score(segmented_project)

    screen(segmented_project, monkeypatch, "33331111", force=True)
    rescored = bad_score(segmented_project)

    assert (rescored["processed"], rescored["skipped"]) == (SPECIMENS, 0)


def test_segment_and_stored_metrics_share_one_recipe(segmented_project):
    """A mixed recipe builds the segment for the segment-input metrics only."""
    traits(segmented_project)
    metric = cf.derived(width_ratio, [cf.body_length(), cf.max_width()], from_run="traits")
    result = cf.run_metrics(
        segmented_project, run_name="mixed", metrics=[cf.mask_area(), metric], visualize=False
    )["organism"]
    assert result["processed"] == SPECIMENS


def test_a_derived_metric_can_read_its_own_runs_earlier_metrics(segmented_project):
    """One pass, one recipe: the value and what is derived from it go stale together."""
    ratio = cf.derived(width_ratio, [cf.body_length(), cf.max_width()])
    result = cf.run_metrics(
        segmented_project,
        run_name="traits",
        transforms=[cf.crop_to_mask()],
        metrics=[cf.body_length(), cf.max_width(), ratio],
        visualize=False,
    )["organism"]
    assert result["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False)
    expected = exported["traits__organism__max_width"] / exported["traits__organism__body_length"]
    assert exported["traits__organism__width_ratio"].tolist() == pytest.approx(expected.tolist())


def test_colour_presence_in_the_run_that_measures_the_fractions(segmented_project):
    bins = cf.threshold_fractions(
        [cf.color_threshold("bright", lch_l=(50, None)), cf.color_threshold("dark", lch_l=(None, 50))],
        unmatched=True,
        name="color_bins",
    )
    cf.run_metrics(
        segmented_project,
        run_name="colour",
        metrics=[bins, cf.color_presence(bins, min_fraction=0.5)],
        visualize=False,
    )

    exported = cf.export_metrics(segmented_project, path=False)
    bright = exported["colour__organism__color_bins__bright"]
    assert (exported["colour__organism__color_presence__bright_present"] == (bright >= 0.5)).all()
    assert exported["colour__organism__color_presence__n_colors_present"].between(0, 2).all()
    assert "colour__organism__color_presence__ranked_color_2" in exported


def test_a_derived_metric_can_name_its_runs_earlier_metrics(segmented_project):
    """Named or held, it is one recipe: the second form finds the first's work done."""

    def run(ratio):
        return cf.run_metrics(
            segmented_project,
            run_name="traits",
            transforms=[cf.crop_to_mask()],
            metrics=[cf.body_length(), cf.max_width(), ratio],
            visualize=False,
        )["organism"]

    assert run(cf.derived(width_ratio, ["body_length", "max_width"]))["processed"] == SPECIMENS
    again = run(cf.derived(width_ratio, [cf.body_length(), cf.max_width()]))
    assert again["processed"] == 0 and again["skipped"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False)
    expected = exported["traits__organism__max_width"] / exported["traits__organism__body_length"]
    assert exported["traits__organism__width_ratio"].tolist() == pytest.approx(expected.tolist())


def test_colour_presence_finds_the_fractions_measured_before_it(segmented_project):
    def bins():
        return cf.threshold_fractions(
            [cf.color_threshold("bright", lch_l=(50, None)), cf.color_threshold("dark", lch_l=(None, 50))],
            unmatched=True,
            name="color_bins",
        )

    def run(metrics):
        return cf.run_metrics(segmented_project, run_name="colour", metrics=metrics, visualize=False)[
            "organism"
        ]

    assert run([bins(), cf.color_presence(min_fraction=0.5)])["processed"] == SPECIMENS
    held = bins()
    again = run([held, cf.color_presence(held, min_fraction=0.5)])
    assert again["processed"] == 0 and again["skipped"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False)
    bright = exported["colour__organism__color_bins__bright"]
    assert (exported["colour__organism__color_presence__bright_present"] == (bright >= 0.5)).all()


@pytest.mark.parametrize(
    "metrics",
    [
        lambda ratio: [ratio, cf.body_length(), cf.max_width()],  # listed before its features
        lambda ratio: [cf.body_length(), ratio],  # one feature missing
    ],
)
def test_a_same_run_feature_must_be_listed_first(segmented_project, metrics):
    ratio = cf.derived(width_ratio, [cf.body_length(), cf.max_width()])
    with pytest.raises(ValueError, match="listed before it"):
        cf.run_metrics(segmented_project, run_name="traits", metrics=metrics(ratio), visualize=False)
    assert cf.export_metrics(segmented_project, path=False).empty
