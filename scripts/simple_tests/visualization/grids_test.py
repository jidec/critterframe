"""
The two grid layouts, and the sampling that feeds them, with no project
involved.

Draws a handful of fake panels of deliberately mismatched sizes -- which is the
case that matters, since a crop, a rotation, and a mask panel are never the same
shape -- lays them out both ways, and writes the results somewhere you can open
them.

Run from the repo root:
    python scripts/simple_tests/visualization/grids_test.py
"""

import logging
import os
import tempfile

import cv2
import numpy as np

from critterframe.selectionhelpers import sample_occurrences
from critterframe.visualization.grids import comparison_grid, image_grid

logging.basicConfig(level=logging.INFO)

OUT_DIR = os.path.join(tempfile.gettempdir(), "critterframe_grids_test")
os.makedirs(OUT_DIR, exist_ok=True)

IDS = [f"specimen{index:03d}" for index in range(12)]


def panel(width, height, color, text):
    """One fake diagnostic panel: a blob on a dark frame, captioned like a real one."""
    image = np.full((height, width, 3), 40, np.uint8)
    cv2.ellipse(image, (width // 2, height // 2),
                (width // 3, height // 4), 20, 0, 360, color, -1)
    cv2.putText(image, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255), 1)
    return image


print("== sampling ==")
print("n=5 :", sample_occurrences(IDS, 5))
print("n=5 :", sample_occurrences(IDS, 5), "  <- identical: the sample is stable")
print("n=99:", len(sample_occurrences(IDS, 99)), "of", len(IDS),
      "  <- asking for more than there are returns them all")

print("\n== image grid: one view, many occurrences ==")
grid = image_grid(
    [panel(280 + 20 * index, 200 + 10 * index, (200, 180, 150), f"{3900 + index}px")
     for index in range(12)],
    labels=IDS, title="segments / organism / n=12 / abc123", columns=5)
cv2.imwrite(os.path.join(OUT_DIR, "image_grid.png"), grid)
print("  ", grid.shape, "-> image_grid.png")

print("\n== comparison grid: many views, many occurrences ==")
stages = ["original", "cropped", "oriented", "mask"]
rows = []
for index in range(5):
    rows.append([
        panel(280, 200, (200, 180, 150), "original"),
        panel(120, 160, (200, 180, 150), "cropped"),
        panel(160, 120, (180, 200, 150), "oriented"),
        # A ragged row on purpose: one occurrence's last stage is missing, and
        # the grid must leave a hole rather than shifting its row left.
        panel(120, 160, (255, 255, 255), "mask") if index != 2 else None,
    ])
comparison = comparison_grid(rows, column_titles=stages, row_labels=IDS[:5],
                              title="traits / organism / n=5 / def456")
cv2.imwrite(os.path.join(OUT_DIR, "comparison_grid.png"), comparison)
print("  ", comparison.shape, "-> comparison_grid.png")
print("   row 3's last cell is empty on purpose -- a missing stage leaves a hole")

print(f"\nwritten to {OUT_DIR}")
