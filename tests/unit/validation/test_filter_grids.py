"""
The pictures behind a filter calibration.

Recall and cost say how well a filter set does; they don't say WHICH organisms
it missed or threw away, and that is what tells you what score is missing or
whether a cutoff sits somewhere sensible. So a calibration and an audit draw the
labelled items themselves: by outcome, in order along each score, per filter,
per category.

Every grid is drawn from the labelled rows and capped, so it stays bounded
whatever the project's size, and a grid with nothing to show is not written.
"""

import numpy as np
import pytest

import critterframe as cf
from critterframe.project import paths
from critterframe.core.recipes import Recipe
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run
from critterframe.validation.filter_grids import STRIP_CELLS, thin_ranks
from critterframe.validation.filters import audit_filters, get_validated_filters
from critterframe.visualization.panels import bordered

pytestmark = pytest.mark.slow

IDS = [f"specimen{index}" for index in range(8)]
SCORE = "qc__organism__score"
DEPTH = "qc__organism__depth"


def store(project_path, run_name, metric_name, values):
    recipe = Recipe("metric", run_name, [cf.body_length()], part="organism")
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [
            make_metric_row(occurrence_id, "organism", metric_name, value)
            for occurrence_id, value in values.items()
        ],
    )


def labelled(project_path):
    """
    Eight screened organisms, three of them bad:

      specimen  0    1    2     3    4..7
      label     bad  bad  good  bad  good
      score     1    2    3     4    5..8     low is bad, and misses specimen3
      depth     8    7    3     6    5,4,2,1  high is bad, and catches all three
      cluster   0    0    1     1    1
    """
    store(project_path, "qc", "score", dict(zip(IDS, range(1, 9))))
    store(project_path, "qc", "depth", dict(zip(IDS, [8, 7, 3, 6, 5, 4, 2, 1])))
    store(
        project_path,
        "clusters",
        "cluster",
        {occurrence_id: {"cluster_id": 0 if index < 2 else 1} for index, occurrence_id in enumerate(IDS)},
    )
    store(
        project_path,
        "quality",
        "quality",
        dict(zip(IDS, ["bad", "bad", "good", "bad", "good", "good", "good", "good"])),
    )


def grids_written(project_path, stem):
    """The suffixes of the image files a report wrote, e.g. {"outcomes", "qc__score__strip"}."""
    directory = paths.pipeline_dir(project_path)
    if not directory.exists():
        return set()
    # A report's files are <stem>_<16-hex identity>__<suffix>.png.
    prefix = len(stem) + len("_") + 16 + len("__")
    return {path.stem[prefix:] for path in directory.glob(f"{stem}_*.png")}


def calibrate(project_path, **kwargs):
    return get_validated_filters(
        project_path,
        {"qc": {"score": "below"}, "clusters": {"cluster__cluster_id": "category"}},
        annotation_run="quality",
        label_metric="quality",
        bad_labels=["bad"],
        max_fpr=0.0,
        min_labelled=2,
        **kwargs,
    )


def test_a_calibration_draws_the_items_it_was_judged_on(segmented_project):
    """
    score >= 3 removes specimen0 and specimen1 and keeps specimen3, so there is
    a miss to show and no good row lost. Both filters remove the same two rows,
    so neither removes anything alone.
    """
    labelled(segmented_project)
    filters = calibrate(segmented_project)
    assert sorted(filters) == ["clusters__organism__cluster__cluster_id", SCORE]

    written = grids_written(segmented_project, "filters__qc+clusters__vs__quality")
    assert {"outcomes", "misses", "qc__score__strip", "clusters__cluster__cluster_id__categories"} <= written
    assert "cost" not in written  # no good row was removed
    assert "only_here" not in written  # nothing removed by one filter alone


def test_visualize_false_draws_nothing(segmented_project):
    labelled(segmented_project)
    calibrate(segmented_project, visualize=False)
    assert not paths.pipeline_dir(segmented_project).exists()


def test_an_audit_draws_outcomes_cost_and_what_each_filter_alone_removes(segmented_project):
    """
    A hand-written set: score >= 4 costs specimen2, a good one, and depth <= 5
    is alone in removing specimen3. An audit sweeps nothing, so it has no strip.
    """
    labelled(segmented_project)
    audit_filters(
        segmented_project,
        {SCORE: (">=", 4), DEPTH: ("<=", 5)},
        "quality",
        label_metric="quality",
        bad_labels=["bad"],
    )

    written = grids_written(segmented_project, "audit_filters__quality")
    assert {"outcomes", "cost", "only_here"} <= written
    assert "misses" not in written  # depth catches all three
    assert not any(name.endswith("__strip") for name in written)


def test_a_grid_is_a_picture_of_the_right_size(segmented_project):
    """Four outcome rows, each a heading cell and up to ten captioned cutouts."""
    import cv2

    labelled(segmented_project)
    calibrate(segmented_project)
    path = next(paths.pipeline_dir(segmented_project).glob("filters__*__outcomes.png"))
    grid = cv2.imread(str(path))

    assert grid is not None and grid.ndim == 3
    assert grid.shape[0] > 4 * 160  # four rows of cells
    assert grid.shape[1] >= 2 * 160  # a heading and at least one item


def test_a_project_with_no_images_still_gets_its_filters_and_charts(metadata_project):
    """The grids are the visual check, not the result: without pictures the numbers still come back."""
    labelled(metadata_project)
    filters = calibrate(metadata_project)

    assert SCORE in filters
    written = grids_written(metadata_project, "filters__qc+clusters__vs__quality")
    assert "together" in written
    assert "outcomes" not in written


# ---------------------------------------------------------------------------
# The pieces, with no project
# ---------------------------------------------------------------------------


def test_a_short_strip_shows_everything():
    assert thin_ranks(7, STRIP_CELLS) == list(range(7))


def test_a_long_strip_is_thinned_evenly_and_keeps_its_ends():
    ranks = thin_ranks(1000, 40)
    assert len(ranks) <= 40
    assert ranks == sorted(ranks)
    assert ranks[0] == 0 and ranks[-1] == 999


def test_the_items_beside_the_cutoff_are_always_shown():
    """
    The cutoff is the point of the strip, and thinning by rank alone would
    usually skip the two items it falls between.
    """
    ranks = thin_ranks(1000, 40, keep=(516, 517))
    assert {516, 517} <= set(ranks)
    assert len(ranks) <= 40


def test_a_kept_rank_outside_the_strip_is_ignored():
    """A cutoff past either end has only one neighbour."""
    assert thin_ranks(5, 40, keep=(-1, 0)) == [0, 1, 2, 3, 4]
    assert set(thin_ranks(100, 10, keep=(99, 100))) >= {99}


def test_a_border_frames_a_cell_without_touching_its_middle():
    image = np.full((40, 60, 3), 7, np.uint8)
    framed = bordered(image, (0, 0, 255), 5)

    assert framed.shape == image.shape
    assert (framed[:5] == (0, 0, 255)).all() and (framed[:, -5:] == (0, 0, 255)).all()
    assert np.array_equal(framed[5:-5, 5:-5], image[5:-5, 5:-5])
    assert (image == 7).all()  # the original is not drawn on
