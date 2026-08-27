"""
What a run leaves behind for a person to look at.

Two kinds of picture live under `visualizations/`, and which one a picture
belongs to is decided by who it is FOR, not by what is in it:

  `pipeline/` -- sample-level summaries of how processing BEHAVED. One grid per
  run, whatever the run's kind. There is no per-occurrence file mode: a
  10,000-occurrence run cannot be inspected as 10,000 files, so every form of
  `visualize=` samples.

  `products/` -- assets deliberately materialized for downstream use, one file
  per occurrence-part, named so a directory listing joins back to the export.

The property that surprises people, and the one worth pinning: a grid can only
show work that HAPPENED. A fully cached rerun writes none, because there were no
panels to collect. That is what `force=True` is for.
"""

import pytest

import critterframe as cf
from critterframe.project import paths
from helpers.models import ThresholdModel

pytestmark = pytest.mark.slow


def grids(project_path):
    directory = paths.pipeline_dir(project_path)
    return sorted(path.name for path in directory.glob("*.jpg")) \
        if directory.exists() else []


def segment(project_path, **kwargs):
    return cf.run_segments(project_path, steps=[cf.segment(ThresholdModel())],
                           **kwargs)


def measure(project_path, **kwargs):
    return cf.run_metrics(project_path, run_name="traits",
                          transforms=[cf.remove_appendages(), cf.orient()],
                          metrics=[cf.body_length()], **kwargs)


# ---------------------------------------------------------------------------
# pipeline/
# ---------------------------------------------------------------------------


def test_a_segmentation_run_writes_one_grid(image_project):
    segment(image_project, visualize=4)
    assert len(grids(image_project)) == 1


def test_the_grid_is_named_for_the_run_and_its_recipe(image_project):
    """
    So a changed recipe writes a second sheet beside the first rather than
    overwriting the evidence of what the old one did.
    """
    segment(image_project, visualize=4)
    first = grids(image_project)[0]

    cf.run_segments(image_project, steps=[cf.segment(ThresholdModel(erode=2))],
                    visualize=4)
    written = grids(image_project)

    assert first.startswith("segments_")
    assert len(written) == 2


def test_a_metric_run_writes_a_grid_too(segmented_project):
    """
    Segmentation and metric runs are the same object here -- a stage per
    column, an occurrence per row -- so `visualize=` means the same thing in
    both.
    """
    measure(segmented_project, visualize=3)
    assert len(grids(segmented_project)) == 1


def test_a_fully_cached_rerun_writes_no_grid(segmented_project):
    """
    A grid can only show work that happened, and after a cached rerun there was
    none. Not a bug and not silent -- the run reports skipped=8.
    """
    measure(segmented_project, visualize=3)
    before = grids(segmented_project)

    result = measure(segmented_project, visualize=3)["organism"]
    assert result["skipped"] == 8
    assert grids(segmented_project) == before


def test_force_makes_the_work_happen_and_the_grid_with_it(segmented_project):
    measure(segmented_project, visualize=3)
    written = paths.pipeline_dir(segmented_project) / grids(segmented_project)[0]
    first_size = written.stat().st_size

    measure(segmented_project, force=True, visualize=6)
    assert written.stat().st_size != first_size     # same file, more specimens


def test_visualization_is_off_by_default_for_nothing_and_on_for_true(
        image_project):
    segment(image_project, visualize=False)
    assert grids(image_project) == []

    segment(image_project, force=True, visualize=True)
    assert len(grids(image_project)) == 1


def test_named_occurrences_can_be_followed_instead_of_a_sample(segmented_project):
    """
    How you follow one known-difficult specimen through a recipe rather than
    hoping the sample happens to include it.
    """
    measure(segmented_project, visualize=["specimen0", "specimen3"])
    assert len(grids(segmented_project)) == 1


def test_a_multi_part_run_writes_one_grid_per_part(segmented_project):
    """
    And the part is in the filename only when it isn't the default one, so a
    plain whole-organism run keeps the obvious name.
    """
    cf.run_segments(segmented_project, run_name="parts",
                    shared_steps=[cf.remove_background()],
                    from_part="organism",
                    outputs={"core": [cf.segment(ThresholdModel(cutoff=120))],
                             "edge": [cf.segment(ThresholdModel(cutoff=90))]},
                    visualize=3)

    written = grids(segmented_project)
    assert any("__core_" in name for name in written)
    assert any("__edge_" in name for name in written)


def test_a_run_always_contributes_one_panel_of_its_own(segmented_project):
    """
    Even when nothing in the recipe draws: a segmentation run adds the mask it
    settled on, and a metric run adds the measured segment with its values.
    Drawn by the RUN rather than left to the operations, because that is the
    one view that always exists -- and a recipe of operations with no
    visualize() of their own would otherwise put nothing on the sheet at
    exactly the moment you most want to look.
    """
    cf.run_metrics(segmented_project, run_name="areas",
                   metrics=[cf.mask_fraction()], visualize=3)
    assert len(grids(segmented_project)) == 1


def test_the_sample_is_the_same_specimens_across_two_recipes(segmented_project):
    """
    Which is what lets two versions of a recipe be compared cell by cell. The
    sheets differ; the specimens on them do not.
    """
    from critterframe.visualization.pipeline import resolve_sample

    ids = [f"specimen{index}" for index in range(8)]
    assert resolve_sample(ids, 4) == resolve_sample(ids, 4)


# ---------------------------------------------------------------------------
# products/
# ---------------------------------------------------------------------------


def test_a_render_writes_one_file_per_occurrence_part(segmented_project):
    summary = cf.render_segments(segmented_project, "plates",
                                 transforms=[cf.remove_background(),
                                             cf.crop_to_mask(pad=0.1)])
    written = sorted(summary["directory"].glob("*.png"))

    assert len(written) == 8
    assert {path.stem for path in written} == {f"specimen{index}"
                                               for index in range(8)}


def test_products_and_pipeline_sheets_do_not_share_a_folder(segmented_project):
    """
    The two contracts are separate directories precisely so which one a picture
    belongs to is decided by where it lands.
    """
    measure(segmented_project, visualize=3)
    summary = cf.render_segments(segmented_project, "plates",
                                 transforms=[cf.remove_background()])

    assert summary["directory"].parent == paths.products_dir(segmented_project)
    assert paths.pipeline_dir(segmented_project).exists()
    assert not list(paths.products_dir(segmented_project).glob("*.jpg"))
