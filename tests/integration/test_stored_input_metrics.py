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
    return cf.run_metrics(project_path, run_name="traits",
                          metrics=[cf.body_length(), cf.max_width()], **kwargs)["organism"]


def shape(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    metric = cf.derived(width_ratio, [cf.body_length(), cf.max_width()],
                        from_run="traits")
    return cf.run_metrics(project_path, metrics=[metric], **kwargs)["organism"]


def test_a_derived_value_stores_and_exports_like_any_metric(segmented_project):
    traits(segmented_project)
    assert shape(segmented_project)["processed"] == SPECIMENS

    exported = cf.export_metrics(segmented_project, path=False)
    expected = (exported["traits__organism__max_width"]
                / exported["traits__organism__body_length"])
    assert exported["width_ratio__organism__width_ratio"].tolist() == pytest.approx(
        expected.tolist())


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


def test_segment_and_stored_metrics_share_one_recipe(segmented_project):
    """A mixed recipe builds the segment for the segment-input metrics only."""
    traits(segmented_project)
    metric = cf.derived(width_ratio, [cf.body_length(), cf.max_width()],
                        from_run="traits")
    result = cf.run_metrics(segmented_project, run_name="mixed",
                            metrics=[cf.mask_area(), metric],
                            visualize=False)["organism"]
    assert result["processed"] == SPECIMENS
