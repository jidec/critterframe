"""
Comparing masks against a reference. Nothing here persists anything.

The word "reference" is deliberate and is the thing to keep straight: a
reference is whatever you chose to compare against, and calling it ground truth
would assert the answer that validation exists to measure. A hand-corrected mask
is a reference; so is a second model's output; so is a mask from a slower,
better pipeline.

The arithmetic itself (`maskops.mask_iou`, `mask_coverage`) is tested in
tests/unit/test_maskops.py; what is checked here is the comparison a project
makes with it.
"""

import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from helpers.models import ThresholdModel


def a_mask(shape=(100, 100), box=(slice(20, 60), slice(20, 60))):
    mask = np.zeros(shape, bool)
    mask[box] = True
    return mask


# ---------------------------------------------------------------------------
# validate_masks
# ---------------------------------------------------------------------------


@pytest.fixture
def with_reference(segmented_project):
    """
    A project whose canonical masks came from one segmenter and whose reference
    masks came from a stricter one -- the ordinary validation setup, where the
    two genuinely disagree.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, visualize=False)
    return segmented_project


@pytest.mark.slow
def test_validation_compares_the_two_tables(with_reference):
    scores = cf.validate_masks(with_reference, visualize=False)

    assert len(scores) == 8
    # Indexed by occurrence so a caller can look one up, sort by agreement, or
    # join it onto an export without renaming anything.
    assert scores.index.name == "occurrence_id"
    assert scores.columns.tolist() == ["iou"]
    assert (scores["iou"] > 0).all()
    assert (scores["iou"] < 1).all()       # a stricter segmenter really differs


@pytest.mark.slow
def test_an_occurrence_with_no_reference_is_left_out(segmented_project):
    """
    Reference masks are expensive -- somebody drew them -- so a project has
    them for a handful of occurrences, and validation reports on that handful
    rather than on the whole project.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, limit=3, visualize=False)

    assert len(cf.validate_masks(segmented_project, visualize=False)) == 3


@pytest.mark.slow
def test_validation_persists_nothing(with_reference):
    """
    All comparison, nothing stored. A validation score is a judgement about a
    method rather than a property of an organism, and storing it as a metric
    would put it in the trait table.
    """
    from critterframe.records.runs import load_runs

    before_runs = len(load_runs(with_reference))
    before_masks = len(mask_records.load_masks(with_reference))

    cf.validate_masks(with_reference, visualize=False)

    assert len(load_runs(with_reference)) == before_runs
    assert len(mask_records.load_masks(with_reference)) == before_masks
    assert len(cf.export_metrics(with_reference)) == 0


@pytest.mark.slow
def test_transforms_are_applied_to_both_sides(with_reference):
    """
    Otherwise the comparison would measure the transform rather than the
    disagreement -- which is why the chain is applied to the reference too.
    """
    plain = cf.validate_masks(with_reference, visualize=False)
    cleaned = cf.validate_masks(with_reference,
                                transforms=[cf.remove_appendages()],
                                visualize=False)
    assert len(plain) == len(cleaned)


@pytest.mark.slow
def test_a_project_with_no_reference_masks_has_nothing_to_validate(
        segmented_project, caplog):
    with caplog.at_level("WARNING"):
        scores = cf.validate_masks(segmented_project, visualize=False)
    assert len(scores) == 0


# ---------------------------------------------------------------------------
# validate_masks(steps=...) -- computed live, no prior run_segments() needed
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_steps_computes_predictions_live(segmented_project):
    """
    A candidate recipe checked against the reference set with no prior
    run_segments() pass -- the whole point of `steps=`.
    """
    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, visualize=False)

    scores = cf.validate_masks(segmented_project,
                               steps=[cf.segment(ThresholdModel())],
                               visualize=False)

    assert len(scores) == 8
    assert scores.index.name == "occurrence_id"
    assert scores.columns.tolist() == ["iou"]
    assert (scores["iou"] > 0).all()
    assert (scores["iou"] < 1).all()       # erode=2 reference genuinely differs


@pytest.mark.slow
def test_steps_persists_nothing(segmented_project):
    from critterframe.records.runs import load_runs

    cf.run_segments(segmented_project, run_name="by_hand",
                    steps=[cf.segment(ThresholdModel(erode=2))],
                    reference=True, visualize=False)

    before_runs = len(load_runs(segmented_project))
    before_masks = len(mask_records.load_masks(segmented_project))

    cf.validate_masks(segmented_project, steps=[cf.segment(ThresholdModel())],
                      visualize=False)

    assert len(load_runs(segmented_project)) == before_runs
    assert len(mask_records.load_masks(segmented_project)) == before_masks
    assert len(cf.export_metrics(segmented_project)) == 0


@pytest.mark.slow
def test_steps_includes_occurrences_with_no_canonical_mask(with_reference):
    """
    Population-selection proof: an occurrence with a reference mask but no
    canonical mask is excluded by default and included under `steps=`.
    """
    from critterframe.project import paths
    from critterframe.storage.tables import load_table, write_table

    canonical_path = paths.masks_path(with_reference)
    canonical = load_table(canonical_path)
    orphan_id = canonical.iloc[0]["occurrence_id"]
    write_table(canonical[canonical["occurrence_id"] != orphan_id], canonical_path)

    default_scores = cf.validate_masks(with_reference, visualize=False)
    live_scores = cf.validate_masks(with_reference,
                                    steps=[cf.segment(ThresholdModel())],
                                    visualize=False)

    assert orphan_id not in default_scores.index
    assert orphan_id in live_scores.index


# ---------------------------------------------------------------------------
# validate_masks(visualize=..., label=...)
# ---------------------------------------------------------------------------


class DrawsItsOwn(ThresholdModel):
    """A model with its own diagnostic panel, the same shape test_segment_op.py
    uses to prove `visualize()` fires through run_segments."""

    def visualize(self, segment, image, mask, score, info):
        segment.emit_panel(np.zeros_like(image), "custom")


def _sidecars(project_path, prefix):
    import json

    from critterframe.project import paths

    return [json.loads(path.read_text(encoding="utf-8"))
            for path in paths.pipeline_dir(project_path).glob(f"{prefix}_*.report.json")]


@pytest.mark.slow
def test_visualize_writes_a_worst_first_grid_and_a_histogram(with_reference):
    """One bounded sheet of the lowest scores, not one loose file per occurrence."""
    scores = cf.validate_masks(with_reference, visualize=2)

    [record] = _sidecars(with_reference, "validate_masks")
    shown = [entry["value"] for entry in record["shown"]]
    assert shown == sorted(scores["iou"])[:2]
    assert any(name.endswith(".jpg") for name in record["files"])
    assert any(name.endswith("__iou.png") for name in record["files"])


@pytest.mark.slow
def test_visualize_surfaces_a_steps_operation_own_diagnostic_panel(with_reference):
    """
    A model's own visualize() (e.g. GroundedSAM2's box overlay) needs a sink on
    the live Segment `steps` runs against, or it never reaches disk here.
    """
    import cv2

    from critterframe.project import paths

    cf.validate_masks(with_reference, steps=[cf.segment(DrawsItsOwn())],
                      visualize=1, limit=1)

    [record] = _sidecars(with_reference, "validate_masks")
    [grid] = [name for name in record["files"] if name.endswith(".jpg")]
    height, width = cv2.imread(str(paths.pipeline_dir(with_reference) / grid)).shape[:2]
    assert width > height   # "custom" and "compare" side by side, not one lone cell


@pytest.mark.slow
def test_label_names_the_output_and_candidates_never_overwrite(with_reference):
    """A sweep's candidates each keep their own sheet, told apart by label and by hash."""
    cf.validate_masks(with_reference, steps=[cf.segment(ThresholdModel())], label="a")
    cf.validate_masks(with_reference, steps=[cf.segment(ThresholdModel(erode=2))], label="b")

    assert len(_sidecars(with_reference, "validate_masks__a")) == 1
    assert len(_sidecars(with_reference, "validate_masks__b")) == 1


@pytest.mark.slow
def test_visualize_false_writes_nothing(with_reference):
    from critterframe.project import paths

    cf.validate_masks(with_reference, visualize=False)
    assert not paths.visualizations_dir(with_reference).exists()


# ---------------------------------------------------------------------------
# validate_masks(metric=..., reference_part=...)
# ---------------------------------------------------------------------------


def _set_masks(project_path, occurrence_id, organism=None, body=None):
    """
    Overwrite one occurrence's canonical 'organism' mask and/or reference
    'body' mask directly, for full control over the shapes a test compares --
    the same direct-table-write pattern
    test_steps_includes_occurrences_with_no_canonical_mask above uses.
    """
    if organism is not None:
        mask_records.save_masks(
            project_path, [mask_records.make_mask_row(occurrence_id, organism,
                                                       part="organism")])
    if body is not None:
        mask_records.save_masks(
            project_path, [mask_records.make_mask_row(occurrence_id, body,
                                                       part="body")],
            reference=True)


def test_metric_defaults_to_iou(with_reference):
    """Backward compatible: no metric= means the original iou column."""
    scores = cf.validate_masks(with_reference, visualize=False)
    assert scores.columns.tolist() == ["iou"]


def test_an_unknown_metric_is_rejected(segmented_project):
    with pytest.raises(ValueError):
        cf.validate_masks(segmented_project, metric="recall", visualize=False)


def test_reference_part_compares_against_a_different_named_part(segmented_project):
    """
    'organism' predicted against a reference stored under 'body' instead of
    'organism' -- e.g. a merged part from merge_masks() -- with no new
    function needed.
    """
    occurrence_id = mask_records.load_masks(segmented_project).iloc[0]["occurrence_id"]
    _set_masks(segmented_project, occurrence_id,
              organism=a_mask(box=(slice(0, 100), slice(0, 100))),
              body=a_mask(box=(slice(20, 60), slice(20, 60))))

    default_scores = cf.validate_masks(segmented_project, part="organism",
                                       visualize=False)
    body_scores = cf.validate_masks(segmented_project, part="organism",
                                    reference_part="body", visualize=False)

    assert occurrence_id not in default_scores.index  # no "organism" reference
    assert occurrence_id in body_scores.index


def test_coverage_metric_ignores_predicted_area_outside_the_reference(segmented_project):
    """
    End-to-end version of what mask_coverage already proves in isolation: an
    organism mask several times larger than the reference "body" part, but
    that fully contains it, scores 1.0 under metric="coverage" and less than
    1.0 under the default IoU.
    """
    occurrence_id = mask_records.load_masks(segmented_project).iloc[0]["occurrence_id"]
    _set_masks(segmented_project, occurrence_id,
              organism=a_mask(box=(slice(0, 100), slice(0, 100))),
              body=a_mask(box=(slice(20, 60), slice(20, 60))))

    iou_scores = cf.validate_masks(segmented_project, part="organism",
                                   reference_part="body", visualize=False)
    coverage_scores = cf.validate_masks(segmented_project, part="organism",
                                        reference_part="body", metric="coverage",
                                        visualize=False)

    assert coverage_scores.columns.tolist() == ["coverage"]
    assert coverage_scores.loc[occurrence_id, "coverage"] == 1.0
    assert iou_scores.loc[occurrence_id, "iou"] < 1.0


# ---------------------------------------------------------------------------
# validate_masks(parts=...)
# ---------------------------------------------------------------------------


def test_parts_returns_one_frame_per_part(segmented_project):
    """
    A project that segments three body parts validates three, and getting one
    concatenated frame back would lose which score belonged to which part.
    """
    occurrence_id = mask_records.load_masks(segmented_project).iloc[0]["occurrence_id"]
    for part in ("organism", "body"):
        mask_records.save_masks(
            segmented_project,
            [mask_records.make_mask_row(occurrence_id,
                                        a_mask(box=(slice(0, 100), slice(0, 100))),
                                        part=part)])
        mask_records.save_masks(
            segmented_project,
            [mask_records.make_mask_row(occurrence_id,
                                        a_mask(box=(slice(0, 100), slice(0, 100))),
                                        part=part)],
            reference=True)

    results = cf.validate_masks(segmented_project, parts=["organism", "body"],
                                visualize=False)

    assert set(results) == {"organism", "body"}
    assert results["organism"].loc[occurrence_id, "iou"] == 1.0
    assert results["body"].loc[occurrence_id, "iou"] == 1.0


def test_one_part_still_returns_a_bare_frame(with_reference):
    """
    `parts=` is what asks for the map. The single-part call is the common one
    and keeps handing back the frame a caller means to read straight away.
    """
    assert isinstance(cf.validate_masks(with_reference, visualize=False),
                      pd.DataFrame)


def test_parts_and_reference_part_together_are_refused(segmented_project):
    """
    reference_part names one specific pairing -- organism against a merged
    'body' -- so it has no meaning spread across a list of parts, and silently
    comparing every part against one reference would be wrong rather than
    unsupported.
    """
    with pytest.raises(ValueError):
        cf.validate_masks(segmented_project, parts=["organism", "body"],
                          reference_part="body", visualize=False)


@pytest.mark.slow
def test_a_moving_transform_still_compares_the_same_pixels(with_reference):
    """
    A chain of purely SPATIAL transforms cannot change how much two masks
    agree: whatever it does to the pixels, it does to both, and inverting
    each back to the frame the masks were stored in undoes it.

    It does change the agreement if each side is compared in its own
    post-transform frame -- `crop_to_mask` crops each mask to its own extent,
    so the two land in differently-sized frames that get padded and scored
    against misaligned pixels.
    """
    plain = cf.validate_masks(with_reference, visualize=False)
    moved = cf.validate_masks(with_reference,
                              transforms=[cf.crop_to_mask(), cf.orient()],
                              visualize=False)

    assert moved["iou"].tolist() == pytest.approx(plain["iou"].tolist(), abs=0.02)
