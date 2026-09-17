"""
Comparing masks against a reference. Nothing here persists anything.

The word "reference" is deliberate and is the thing to keep straight: a
reference is whatever you chose to compare against, and calling it ground truth
would assert the answer that validation exists to measure. A hand-corrected mask
is a reference; so is a second model's output; so is a mask from a slower,
better pipeline.

`mask_iou` pads to a common shape rather than refusing mismatched inputs,
because the comparison is between masks of the same occurrence and a shape
difference means one of them was stored from a differently-sized frame -- an
answer of "these agree about this much" is more useful there than a crash.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from critterframe.validation.masks import mask_coverage, mask_iou
from helpers.models import ThresholdModel


def a_mask(shape=(100, 100), box=(slice(20, 60), slice(20, 60))):
    mask = np.zeros(shape, bool)
    mask[box] = True
    return mask


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------


def test_identical_masks_agree_completely():
    score, _mask, _reference = mask_iou(a_mask(), a_mask())
    assert score == 1.0


def test_disjoint_masks_agree_about_nothing():
    other = a_mask(box=(slice(70, 90), slice(70, 90)))
    score, _mask, _reference = mask_iou(a_mask(), other)
    assert score == 0.0


def test_partial_overlap_is_intersection_over_union():
    """
    Two 40x40 boxes overlapping in a 20x40 strip: 800 shared out of 2400
    covered.
    """
    other = a_mask(box=(slice(20, 60), slice(40, 80)))
    score, _mask, _reference = mask_iou(a_mask(), other)
    assert score == pytest.approx(800 / 2400)


def test_two_empty_masks_agree():
    """
    Vacuously, and 1.0 is the right answer rather than a division by zero:
    neither found anything, and they do not disagree about where it is.
    """
    score, _mask, _reference = mask_iou(np.zeros((10, 10), bool),
                                        np.zeros((10, 10), bool))
    assert score == 1.0


def test_something_against_nothing_agrees_about_nothing():
    score, _mask, _reference = mask_iou(a_mask(), np.zeros((100, 100), bool))
    assert score == 0.0


def test_masks_of_different_sizes_are_padded_to_compare():
    """
    A shape mismatch means one was stored from a differently-sized frame. "They
    agree about this much" beats a crash, and the padded pair comes back so a
    caller can see what was compared.
    """
    small = a_mask(shape=(80, 80))
    score, padded, padded_reference = mask_iou(small, a_mask())

    assert padded.shape == padded_reference.shape == (100, 100)
    assert score == 1.0


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


@pytest.mark.slow
def test_visualize_surfaces_a_steps_operation_own_diagnostic_panel(with_reference):
    """
    Before wiring a panel_sink into the live Segment `steps` runs against, a
    model's own visualize() (e.g. GroundedSAM2's box overlay) was gated on a
    sink validate_masks never attached, so it could never reach disk here.
    """
    from critterframe.project import paths

    cf.validate_masks(with_reference, steps=[cf.segment(DrawsItsOwn())],
                      visualize=True, limit=1)

    custom_panels = list(paths.visualizations_dir(
        with_reference, "validate_masks/custom").glob("*.png"))
    assert len(custom_panels) == 1


@pytest.mark.slow
def test_label_namespaces_visualization_output(with_reference):
    """
    Two calls comparing different candidates -- a parameter sweep, say --
    must not have the second call's diff panels overwrite the first's.
    """
    from critterframe.project import paths

    scores_a = cf.validate_masks(with_reference, steps=[cf.segment(ThresholdModel())],
                                 visualize=True, label="a")
    scores_b = cf.validate_masks(with_reference,
                                 steps=[cf.segment(ThresholdModel(erode=2))],
                                 visualize=True, label="b")

    a_files = list(paths.visualizations_dir(with_reference, "validate_masks/a").glob("*.png"))
    b_files = list(paths.visualizations_dir(with_reference, "validate_masks/b").glob("*.png"))

    assert len(a_files) == len(scores_a)
    assert len(b_files) == len(scores_b)


@pytest.mark.slow
def test_no_label_keeps_the_flat_default_path(with_reference):
    """Backward compatible: callers that never pass `label` see the same
    `visualizations/validate_masks/` path as before."""
    from critterframe.project import paths

    cf.validate_masks(with_reference, visualize=True, limit=1)

    default_files = list(paths.visualizations_dir(with_reference, "validate_masks").glob("*.png"))
    assert len(default_files) == 1


# ---------------------------------------------------------------------------
# mask_coverage
# ---------------------------------------------------------------------------


def test_identical_masks_have_full_coverage():
    assert mask_coverage(a_mask(), a_mask()) == 1.0


def test_disjoint_masks_have_no_coverage():
    other = a_mask(box=(slice(70, 90), slice(70, 90)))
    assert mask_coverage(a_mask(), other) == 0.0


def test_coverage_is_intersection_over_reference_area_only():
    """
    A 20x40 strip shared out of the reference's own 40x40 -- unlike IoU, the
    denominator is the reference alone, so area the mask covers outside the
    reference never appears in it.
    """
    reference = a_mask(box=(slice(20, 60), slice(40, 80)))
    assert mask_coverage(a_mask(), reference) == pytest.approx(800 / 1600)


def test_extra_predicted_area_outside_the_reference_costs_nothing():
    """
    The property `mask_iou` doesn't have: a mask several times larger than the
    reference, but that fully contains it, still scores 1.0 -- an organism
    mask that also picks up the wings shouldn't be penalized when all that
    matters downstream is that the body is inside it.
    """
    reference = a_mask(box=(slice(20, 60), slice(20, 60)))
    much_bigger = a_mask(box=(slice(0, 100), slice(0, 100)))
    assert mask_coverage(much_bigger, reference) == 1.0

    iou, _mask, _reference = mask_iou(much_bigger, reference)
    assert iou < 1.0


def test_empty_reference_has_vacuous_full_coverage():
    score = mask_coverage(a_mask(), np.zeros((100, 100), bool))
    assert score == 1.0


def test_coverage_pads_masks_of_different_sizes():
    small_reference = a_mask(shape=(80, 80))
    score = mask_coverage(a_mask(), small_reference)
    assert score == 1.0


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
