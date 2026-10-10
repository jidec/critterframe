"""Fixtures the ingest tests share."""

import cv2
import pandas as pd
import pytest

from helpers.synthetic import draw_specimen


@pytest.fixture
def source_csv(tmp_path):
    """A small occurrence export with a determination column to drop on."""
    path = tmp_path / "export.csv"
    pd.DataFrame(
        {
            "detection_id": [1, 2, 3, 4],
            "photo": [f"http://example/{index}.jpg" for index in range(4)],
            "determination_name": ["Noctuidae", "Not Lepidoptera", "Geometridae", "Debris"],
            "captured": ["2024-05-01", "2024-05-02", "bad date", "2024-05-04"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def image_dir(tmp_path):
    """A folder of three drawn specimens."""
    directory = tmp_path / "images"
    directory.mkdir()
    for index in range(3):
        cv2.imwrite(str(directory / f"spec{index}.png"), draw_specimen(index))
    return directory
