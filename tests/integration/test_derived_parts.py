"""
A part cut out of another part, and what happens to it when its upstream moves.

`from_part` is how refinement chains work: a part-specific segmenter starts from
the organism mask rather than rediscovering the organism. The dependency is real
-- `remove_background()` blanks everything outside the upstream mask, so what
the derived segmenter can find is bounded by what the organism mask contained.

The failure this file exists for is invisible from the derived part's own
recipe. Resegmenting the organism does not change the core recipe or its hash,
so a completion check keyed on the recipe alone would report every occurrence as
already done -- leaving core masks cut out of an organism the project no longer
holds, and core measurements taken from those masks, with nothing anywhere
saying so.
"""

import pytest

import critterframe as cf
from critterframe.records import masks as mask_records
from critterframe.recipes import Recipe
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow

SPECIMENS = 8
CORE_AREA = "core_traits__core__area_px"


def segment_core(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_segments(project_path, run_name="core", part="core",
                           from_part="organism",
                           shared_steps=[cf.remove_background()],
                           steps=[cf.segment(ThresholdModel(erode=1))],
                           **kwargs)["core"]


def measure_core(project_path, **kwargs):
    kwargs.setdefault("visualize", False)
    return cf.run_metrics(project_path, run_name="core_traits", part="core",
                          metrics=[cf.mask_area(name="area_px", unit="px2")],
                          **kwargs)["core"]


def resegment_organism(project_path, erode=2):
    return cf.run_segments(project_path,
                           steps=[cf.segment(ThresholdModel(erode=erode))],
                           visualize=False)["organism"]


def test_a_derived_part_gets_its_own_masks(segmented_project):
    assert segment_core(segmented_project)["processed"] == SPECIMENS
    assert sorted(mask_records.parts_present(segmented_project)) == ["core",
                                                                     "organism"]


def test_a_derived_mask_records_which_upstream_it_came_from(segmented_project):
    """
    `from_part` says which part it came from; `source_mask_hash` says WHICH MASK
    of that part -- and only the second can tell you it has gone stale.
    """
    segment_core(segmented_project)
    core = mask_records.load_masks(segmented_project, parts=["core"])
    organism = mask_records.load_masks(segmented_project, parts=["organism"])

    assert set(core["from_part"]) == {"organism"}
    assert set(core["source_mask_hash"]) == set(organism["recipe_hash"])


def test_a_derived_part_is_repeat_aware_like_any_other(segmented_project):
    segment_core(segmented_project)
    assert segment_core(segmented_project)["skipped"] == SPECIMENS

    measure_core(segmented_project)
    assert measure_core(segmented_project)["skipped"] == SPECIMENS


def test_the_derived_part_is_bounded_by_its_upstream(segmented_project):
    """
    What makes the organism a real dependency rather than a formality: the core
    is cut from a background-removed frame, so it cannot be bigger than the
    mask it started from.
    """
    segment_core(segmented_project)
    core = mask_records.load_masks(segmented_project, parts=["core"])
    organism = mask_records.load_masks(segmented_project,
                                       parts=["organism"]).set_index("occurrence_id")

    for row in core.itertuples(index=False):
        assert row.area <= organism.loc[row.occurrence_id, "area"]


def test_resegmenting_the_upstream_makes_the_derived_masks_pending(segmented_project):
    """
    THE test. The core recipe hash has not moved -- nothing about the core
    changed -- and yet all eight are work again, because the thing they were
    cut out of is gone.
    """
    segment_core(segmented_project)
    before = Recipe("segment", "core", [cf.segment(ThresholdModel(erode=1))],
                       part="core", from_part="organism").hash

    resegment_organism(segmented_project)
    after = Recipe("segment", "core", [cf.segment(ThresholdModel(erode=1))],
                      part="core", from_part="organism").hash

    assert before == after
    assert segment_core(segmented_project)["processed"] == SPECIMENS


def test_a_core_measurement_survives_the_upstream_moving_until_the_core_does(
        segmented_project):
    """
    The intermediate state, and worth pinning because it is easy to expect
    otherwise: right after the organism is resegmented, the core's stored
    measurements are still CURRENT.

    They describe the core mask the project holds, and that mask has not
    changed -- it is stale with respect to its upstream, which is a different
    claim, and one only the segmentation run is in a position to make (it is
    the thing that knows what the upstream now is). Until then the export is
    honest about what it holds: these numbers do describe the masks in this
    project.
    """
    segment_core(segmented_project)
    measure_core(segmented_project)
    resegment_organism(segmented_project)

    assert len(cf.export_metrics(segmented_project, parts=["core"])) == SPECIMENS


def test_the_change_propagates_to_the_derived_part_s_metrics(segmented_project):
    """
    One level further down. Re-running the core is what replaces its masks, and
    THAT is what makes the core's own measurements stale -- the core metric
    recipe never changed at all. Two hops from a resegmented organism, with
    nothing in between having a moved recipe hash.
    """
    segment_core(segmented_project)
    measure_core(segmented_project)
    before = cf.export_metrics(segmented_project, parts=["core"]).set_index(
        "occurrence_id")

    resegment_organism(segmented_project)
    assert segment_core(segmented_project)["processed"] == SPECIMENS
    assert len(cf.export_metrics(segmented_project, parts=["core"])) == 0

    assert measure_core(segmented_project)["processed"] == SPECIMENS
    after = cf.export_metrics(segmented_project, parts=["core"]).set_index(
        "occurrence_id")
    specimen = before.index[0]
    assert after.loc[specimen, CORE_AREA] != before.loc[specimen, CORE_AREA]


def test_a_derived_mask_s_identity_is_the_chained_hash(segmented_project):
    """
    Which is what makes the propagation work at any depth: the core's stored
    identity moves when its upstream does, so anything derived from the CORE
    would see the same thing happen in turn.
    """
    segment_core(segmented_project)
    first = mask_records.current_derivation_hashes(segmented_project,
                                                   parts=["core"])
    resegment_organism(segmented_project)
    segment_core(segmented_project)
    second = mask_records.current_derivation_hashes(segmented_project,
                                                    parts=["core"])

    assert set(first) == set(second)
    assert all(first[key] != second[key] for key in first)


def test_an_occurrence_with_no_upstream_mask_is_skipped_with_a_warning(
        image_project, caplog):
    """
    There is nothing to start from. Not a failure and not silent -- the run
    says which occurrences it left alone.
    """
    cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                    limit=3, visualize=False)
    with caplog.at_level("WARNING"):
        result = segment_core(image_project)

    assert result["processed"] == 3
    assert "organism" in caplog.text


def test_the_two_parts_measure_independently(segmented_project):
    """
    Any number of parts per occurrence, each with its own masks, metrics, and
    export columns -- and the organism's values are untouched by the core's.
    """
    segment_core(segmented_project)
    measure_core(segmented_project)
    cf.run_metrics(segmented_project, run_name="traits",
                   metrics=[cf.mask_area(name="area_px", unit="px2")],
                   visualize=False)

    exported = cf.export_metrics(segmented_project)
    assert CORE_AREA in exported.columns
    assert "traits__organism__area_px" in exported.columns
    assert (exported[CORE_AREA] <= exported["traits__organism__area_px"]).all()


@pytest.mark.slow
def test_a_derived_part_is_measured_and_rendered_in_its_upstream_frame(segmented_project):
    """
    A part carved out of the organism crop was segmented against that shared
    frame, so measuring or rendering it has to reproduce the same one rather
    than re-deriving a crop from the part's own, much smaller, mask.
    """
    cf.run_segments(segmented_project, run_name="core", from_part="organism",
                    shared_steps=[cf.crop_to_mask()],
                    outputs={"core": [cf.segment(ThresholdModel(cutoff=120))]},
                    visualize=False)

    framed = cf.run_metrics(segmented_project, run_name="core_framed", part="core",
                            from_part="organism", transforms=[cf.crop_to_mask()],
                            metrics=[cf.mask_area()], visualize=False)["core"]
    own = cf.run_metrics(segmented_project, run_name="core_own", part="core",
                         transforms=[cf.crop_to_mask()],
                         metrics=[cf.mask_area()], visualize=False)["core"]

    assert framed["processed"] == own["processed"] == 8

    from critterframe.export import column_name, metrics_wide

    values = metrics_wide(segmented_project, parts=["core"])
    # Cropping to the organism keeps the whole upstream frame, so the part's
    # area inside it is the same pixels; cropping to the part's own mask makes
    # the mask fill its frame instead.
    framed_col = column_name("core_framed", "core", "mask_area")
    own_col = column_name("core_own", "core", "mask_area")
    assert (values[framed_col] == values[own_col]).all()

    rendered = cf.render_segments(segmented_project, "core_plates", part="core",
                                  from_part="organism",
                                  transforms=[cf.crop_to_mask()], visualize=False)
    assert rendered["core"]["processed"] == 8


@pytest.mark.slow
def test_an_unreliable_operation_is_counted_on_the_run(image_project):
    """
    CLAUDE.md promises `info` reaches the run. Until it did, `unreliable` and
    `degenerate` existed only as text on whichever panels happened to be
    sampled.
    """
    from critterframe.records.runs import load_runs

    summary = cf.run_segments(image_project, steps=[cf.segment(ThresholdModel())],
                              visualize=False)["organism"]
    assert summary["flags"] == {}

    measured = cf.run_metrics(image_project, run_name="shapes",
                              transforms=[cf.orient()],
                              metrics=[cf.body_length()], visualize=False)["organism"]

    runs = load_runs(image_project, name="shapes")
    recorded = runs.iloc[0]["context"].get("flags", {})
    assert recorded == measured["flags"]
