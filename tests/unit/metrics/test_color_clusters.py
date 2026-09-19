"""
color_clusters: a shared palette fit per group, and each organism's share of it.

Tested end to end through run_metrics(), since the palette is fit in prepare()
from pixels pooled across the run's own population.
"""

import cv2
import numpy as np
import pytest

import critterframe as cf
from critterframe.records.metrics import load_metrics
from helpers.models import ThresholdModel
from helpers.synthetic import BACKGROUND, BODY_AXES, BODY_CENTRE, SPECIMEN_SIZE

WARM = (80, 110, 255)     # BGR; both bright enough for ThresholdModel to segment
COOL = (255, 150, 70)


def _two_tone(warm_rows):
    """A body whose first `warm_rows` rows are WARM and the rest COOL."""
    image = np.full((*SPECIMEN_SIZE, 3), BACKGROUND, np.uint8)
    body = np.zeros(SPECIMEN_SIZE, np.uint8)
    cv2.ellipse(body, BODY_CENTRE, BODY_AXES, 0, 0, 360, 255, -1)
    image[body > 0] = COOL
    warm = body > 0
    warm[warm_rows:] = False
    image[warm] = WARM
    return image


@pytest.fixture
def two_tone_project(tmp_path):
    """Four organisms, each part WARM and part COOL in different proportions."""
    directory = tmp_path / "specimens"
    directory.mkdir()
    top = BODY_CENTRE[1] - BODY_AXES[1]
    for index, fraction in enumerate([0.2, 0.4, 0.6, 0.8]):
        warm_rows = top + int(2 * BODY_AXES[1] * fraction)
        cv2.imwrite(str(directory / f"o{index}.png"), _two_tone(warm_rows))

    project = tmp_path / "project"
    cf.ingest_images(project, directory)
    cf.run_segments(project, steps=[cf.segment(ThresholdModel())], visualize=False)
    return project


def _values(project_path, **kwargs):
    cf.run_metrics(project_path, metrics=[cf.color_clusters(**kwargs)], visualize=False)
    values = load_metrics(project_path, metric_names=["color_clusters"])
    return dict(zip(values["occurrence_id"], values["value"]))


@pytest.mark.slow
def test_every_organism_gets_a_share_of_every_palette_colour(two_tone_project):
    values = _values(two_tone_project, n_colors=2, sample_pixels=500)
    assert len(values) == 4
    for value in values.values():
        shares = [value["color_0"], value["color_1"]]
        assert sum(shares) == pytest.approx(1.0)
        assert value["dominant"] == shares.index(max(shares))


@pytest.mark.slow
def test_the_palette_is_shared_so_shares_are_comparable(two_tone_project):
    """One palette for everyone: more WARM rows is more of the same cluster."""
    values = _values(two_tone_project, n_colors=2, sample_pixels=500)
    shares = [values[f"o{index}"]["color_0"] for index in range(4)]
    assert shares == sorted(shares) or shares == sorted(shares, reverse=True)
    assert abs(shares[0] - shares[-1]) > 0.4


def test_an_unknown_color_space_raises():
    with pytest.raises(ValueError, match="color_space"):
        cf.color_clusters(color_space="rgb")
