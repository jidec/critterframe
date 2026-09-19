"""
Nothing to work from is not a failure.

An occurrence with no image yet (ingested, not downloaded) or no upstream mask
yet counts as `no_input`, is never written to the failures ledger, and is
attempted again once the input exists. A failure's retry key -- recipe plus
upstream mask -- doesn't move when an image finally arrives, so recording one
would skip that occurrence forever.
"""

import pandas as pd
import pytest

import critterframe as cf
from critterframe.recipes import Metric
from critterframe.records import failures as failure_records
from critterframe.records import masks as mask_records
from critterframe.records import occurrences as occurrence_records
from critterframe.segmentation.run import _build_recipes
from critterframe.storage.imagestore import ImageStore
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

LATE = "late"


def add_occurrence(project_path, occurrence_id=LATE, with_image=False):
    """One more occurrence, copied from the first; its image only if asked."""
    table = occurrence_records.load_occurrences(project_path)
    extra = table.iloc[[0]].copy()
    first = extra[occurrence_records.ID_COL].iloc[0]
    extra[occurrence_records.ID_COL] = occurrence_id
    occurrence_records.save_occurrences(
        project_path, pd.concat([table, extra], ignore_index=True))
    if with_image:
        give_image(project_path, occurrence_id, like=first)
    return first


def give_image(project_path, occurrence_id, like):
    with ImageStore(project_path) as store:
        store.put(occurrence_id, store.get_bytes(like))


def segment_run(project_path, **kwargs):
    return cf.run_segments(project_path, steps=[cf.segment(ThresholdModel())],
                           visualize=False, **kwargs)["organism"]


def test_a_missing_image_is_no_input_and_is_not_recorded(image_project):
    add_occurrence(image_project)

    result = segment_run(image_project)
    assert (result["no_input"], result["failed"]) == (1, 0)
    assert failure_records.load_failures(image_project, stage="segment").empty


def test_the_image_arriving_later_is_picked_up_without_asking(image_project):
    first = add_occurrence(image_project)
    segment_run(image_project)

    give_image(image_project, LATE, like=first)
    result = segment_run(image_project)
    assert (result["processed"], result["no_input"]) == (1, 0)


def test_a_missing_upstream_mask_is_no_input(segmented_project):
    add_occurrence(segmented_project, with_image=True)   # an image, but no organism mask

    result = cf.run_segments(segmented_project, part="body", from_part="organism",
                             steps=[cf.segment(ThresholdModel())],
                             visualize=False)["body"]
    assert (result["no_input"], result["failed"]) == (1, 0)
    assert result["processed"] == 8


def test_a_missing_image_is_no_input_for_a_metric_too(segmented_project):
    add_occurrence(segmented_project)
    brightness = Metric("brightness", lambda segment: float(segment.image.mean()),
                        requires_mask=False)

    result = cf.run_metrics(segmented_project, metrics=[brightness],
                            visualize=False)["organism"]
    assert (result["no_input"], result["failed"]) == (1, 0)
    assert failure_records.load_failures(segmented_project, stage="metric").empty


def test_a_missing_image_recorded_as_a_failure_before_is_attempted_again(image_project):
    """Ledger rows from before this rule would otherwise match forever."""
    first = add_occurrence(image_project)
    [recipe] = _build_recipes(None, [cf.segment(ThresholdModel())], None, None,
                              "organism", None, False).values()
    failure_records.record_failures(image_project, "segment", [{
        "occurrence_id": LATE, "part": "organism",
        "context_hash": mask_records.derivation_hash(recipe.hash, None),
        "error": "no image in the image store"}])

    give_image(image_project, LATE, like=first)
    result = segment_run(image_project)
    assert result["previously_failed"] == 0
    assert result["processed"] == 9
