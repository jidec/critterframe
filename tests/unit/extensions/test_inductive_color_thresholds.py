"""
InductiveColorThresholdMetric: chroma-gate-then-hue-valley colour categories,
discovered per group rather than chosen ahead of time.

Split like the module itself: the numerical helpers (steps 1-5) are pure
functions over numpy arrays, tested directly with synthetic data whose right
answer is known; the class itself is tested end to end through run_metrics(),
the same way any other group metric earns its plumbing.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

import critterframe as cf
from critterframe.extensions.inat_insects.metrics import inductive_color_thresholds as m
from critterframe.recipes import Segment
from critterframe.records.metrics import load_metrics
from helpers.models import ThresholdModel
from helpers.synthetic import BACKGROUND, BODY_AXES, BODY_CENTRE, SPECIMEN_SIZE


# ---------------------------------------------------------------------------
# _to_lab / _chroma_hue
# ---------------------------------------------------------------------------


def test_to_lab_is_true_range_not_offset():
    """Pure gray must land at a=b=0, not OpenCV's 8-bit-storage +128 -- the
    one detail the whole feature depends on getting right."""
    gray = np.array([[128, 128, 128]], dtype=np.uint8)
    lab = m._to_lab(gray)
    assert lab[0, 1] == pytest.approx(0.0, abs=0.5)
    assert lab[0, 2] == pytest.approx(0.0, abs=0.5)


def test_chroma_hue_of_a_pure_a_axis_pixel():
    lab = np.array([[50.0, 60.0, 0.0]])   # a=60, b=0 -> hue 0, chroma 60
    chroma, hue = m._chroma_hue(lab)
    assert chroma[0] == pytest.approx(60.0)
    assert hue[0] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# _circular_variance
# ---------------------------------------------------------------------------


def test_identical_hues_have_zero_variance():
    assert m._circular_variance(np.full(50, 30.0)) == pytest.approx(0.0, abs=1e-9)


def test_uniformly_scattered_hues_have_variance_near_one():
    hues = np.linspace(0, 360, 1000, endpoint=False)
    assert m._circular_variance(hues) == pytest.approx(1.0, abs=1e-6)


def test_hues_clustered_across_the_wrap_have_low_variance():
    """359, 0, 1 degrees are nearly identical angles -- a naive linear
    variance would call them wildly spread; circular variance must not."""
    hues = np.array([359.0, 0.0, 1.0] * 20)
    assert m._circular_variance(hues) < 0.01


# ---------------------------------------------------------------------------
# _find_chroma_gate
# ---------------------------------------------------------------------------


def test_gate_lands_between_a_noisy_and_a_stable_population():
    rng = np.random.default_rng(0)
    noise_chroma = rng.uniform(0, 10, 2000)
    noise_hue = rng.uniform(0, 360, 2000)
    signal_chroma = rng.uniform(40, 60, 2000)
    signal_hue = rng.normal(45, 2, 2000) % 360

    chroma = np.concatenate([noise_chroma, signal_chroma])
    hue = np.concatenate([noise_hue, signal_hue])
    gate = m._find_chroma_gate(chroma, hue, n_bins=20, min_bin_pixels=20)

    assert 10 < gate < 40


def test_no_trusted_bin_falls_back_to_zero(caplog):
    tiny = np.array([1.0, 2.0, 3.0])
    with caplog.at_level("WARNING"):
        gate = m._find_chroma_gate(tiny, tiny, n_bins=20, min_bin_pixels=50)
    assert gate == 0.0
    assert "no chroma bin had" in caplog.text


# ---------------------------------------------------------------------------
# _hue_kde / _find_hue_valleys / _ranges_from_valleys
# ---------------------------------------------------------------------------


def _contains(start, end, angle):
    if start <= end:
        return start <= angle < end
    return angle >= start or angle < end


def test_three_separated_clusters_recover_three_ranges():
    rng = np.random.default_rng(1)
    hues = np.concatenate([rng.normal(centre, 8, 500) % 360
                           for centre in (30, 150, 270)])
    grid, density = m._hue_kde(hues, bandwidth=10.0)
    valleys = m._find_hue_valleys(grid, density, valley_relative_height=0.5,
                                  max_ranges=8)
    ranges = m._ranges_from_valleys(valleys)

    assert len(ranges) == 3
    for centre in (30, 150, 270):
        assert any(_contains(start, end, centre) for start, end in ranges)


def test_a_single_cluster_gives_one_range_covering_the_whole_circle():
    """
    The lone mode's low-density far side still registers as one valley by
    the local-minimum test, but a single valley isn't a boundary BETWEEN two
    clusters -- there's nothing on its other side. Without the < 2 special
    case in _ranges_from_valleys this would produce a zero-width (x, x)
    range that classifies no pixel at all.
    """
    rng = np.random.default_rng(1)
    hues = rng.normal(90, 10, 1000) % 360
    grid, density = m._hue_kde(hues, bandwidth=10.0)
    valleys = m._find_hue_valleys(grid, density, valley_relative_height=0.5,
                                  max_ranges=8)
    assert m._ranges_from_valleys(valleys) == [(0.0, 360.0)]


def test_two_clusters_across_the_wrap_are_still_found():
    rng = np.random.default_rng(1)
    hues = np.concatenate([rng.normal(0, 8, 500) % 360,
                           rng.normal(180, 8, 500) % 360])
    grid, density = m._hue_kde(hues, bandwidth=10.0)
    valleys = m._find_hue_valleys(grid, density, valley_relative_height=0.5,
                                  max_ranges=8)
    assert len(valleys) == 2


def test_no_valleys_at_all_gives_the_whole_circle():
    assert m._ranges_from_valleys(np.array([])) == [(0.0, 360.0)]


def test_ranges_wrap_correctly():
    assert m._ranges_from_valleys([10.0, 200.0]) == [(10.0, 200.0), (200.0, 10.0)]


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


def test_classify_partitions_every_pixel_exactly_once():
    definition = {"gate": 20.0, "hue_ranges": [(0.0, 180.0), (180.0, 360.0)]}
    hue = np.array([10.0, 190.0, 10.0])
    chroma = np.array([5.0, 5.0, 50.0])   # first two below gate -> achromatic
    masks = m._classify(hue, chroma, definition)

    stacked = np.stack(list(masks.values()))
    assert (stacked.sum(axis=0) == 1).all()          # exactly one label per pixel
    assert masks["achromatic"].tolist() == [True, True, False]
    assert masks["hue_0"].tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# The metric, end to end
# ---------------------------------------------------------------------------

WARM = (80, 110, 255)     # BGR; grayscale ~118, hue ~39 degrees -- segments
COOL = (255, 150, 70)     # BGR; grayscale ~172, hue ~280 degrees -- segments
# both above ThresholdModel's default cutoff=100, and well over 90 degrees
# apart in hue.


def _draw(color):
    image = np.full((*SPECIMEN_SIZE, 3), BACKGROUND, np.uint8)
    cv2.ellipse(image, BODY_CENTRE, BODY_AXES, 15, 0, 360, color, -1)
    return image


@pytest.fixture
def colour_grouped_project(tmp_path):
    """
    Two colour groups of five occurrences each ('warm', 'cool') plus a third,
    undersized group of two -- for the min_group_size fallback -- all
    ingested and segmented, ready for a metric run.
    """
    directory = tmp_path / "specimens"
    directory.mkdir()
    ids, groups = [], []
    for group, color, count in [("warm", WARM, 5), ("cool", COOL, 5),
                                ("tiny", WARM, 2)]:
        for index in range(count):
            occurrence_id = f"{group}{index}"
            cv2.imwrite(str(directory / f"{occurrence_id}.png"), _draw(color))
            ids.append(occurrence_id)
            groups.append(group)

    project = tmp_path / "project"
    metadata = pd.DataFrame({"occurrence_id": ids, "color_group": groups})
    cf.ingest_images(project, directory, metadata=metadata)
    cf.run_segments(project, steps=[cf.segment(ThresholdModel())], visualize=False)
    return project


def _run(project_path):
    return cf.run_metrics(
        project_path,
        metrics=[m.inductive_color_thresholds(group_col="color_group",
                                              min_group_size=3, sample_pixels=500)],
        visualize=False)["organism"]


@pytest.mark.slow
def test_fits_and_scores_every_occurrence(colour_grouped_project):
    result = _run(colour_grouped_project)
    assert result["processed"] == 12


@pytest.mark.slow
def test_each_group_gets_its_own_definition(colour_grouped_project):
    _run(colour_grouped_project)
    values = load_metrics(colour_grouped_project,
                          metric_names=["inductive_color_thresholds"])
    by_occurrence = dict(zip(values["occurrence_id"], values["value"]))

    for occurrence_id in ("warm0", "warm1"):
        assert by_occurrence[occurrence_id]["group"] == "warm"
    for occurrence_id in ("cool0", "cool1"):
        assert by_occurrence[occurrence_id]["group"] == "cool"

    # Solid-colour ellipses have no low-chroma population to gate against, so
    # each of these two groups correctly finds no stabilization point
    # (gate=0.0, logged) and one hue range covering everything actually
    # present -- confirmed by every organism pixel landing in it. (POPULATION,
    # and therefore "tiny" below, pools BOTH colours and so is genuinely
    # bimodal -- this doesn't apply to it, which is exactly what the next
    # test is about.)
    for occurrence_id in ("warm0", "warm1", "cool0", "cool1"):
        assert by_occurrence[occurrence_id]["hue_0"] == pytest.approx(1.0)


@pytest.mark.slow
def test_an_undersized_group_falls_back_to_the_population_definition(colour_grouped_project):
    _run(colour_grouped_project)
    values = load_metrics(colour_grouped_project,
                          metric_names=["inductive_color_thresholds"])
    by_occurrence = dict(zip(values["occurrence_id"], values["value"]))

    # "tiny" has only 2 occurrences, below min_group_size=3 -- both must be
    # scored against the population-wide fallback, reported as group=None
    # (metrics.outliers.POPULATION), the same convention OutlierMetric and
    # ColorClusterMetric already use.
    assert by_occurrence["tiny0"]["group"] is None
    assert by_occurrence["tiny1"]["group"] is None

    # "tiny" is drawn in WARM's colour, but POPULATION pools BOTH colours
    # (genuinely bimodal, unlike warm/cool's own single-colour fits) -- a
    # solid-coloured occurrence scored against it should still land entirely
    # in exactly one of the population's ranges, not split across both.
    hue_values = [value for key, value in by_occurrence["tiny0"].items()
                 if key.startswith("hue_")]
    assert sum(hue_values) == pytest.approx(1.0)
    assert sum(value == pytest.approx(1.0) for value in hue_values) == 1


def test_requires_a_fit_before_scoring():
    metric = m.InductiveColorThresholdMetric()
    image = np.zeros((10, 10, 3), np.uint8)
    mask = np.ones((10, 10), bool)
    with pytest.raises(RuntimeError, match="never fit"):
        metric._score(Segment(image, mask=mask, occurrence_id="x"))
