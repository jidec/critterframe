"""
GroundedSAM2 over a folder of test crops, both prompting paths.

The two paths answer different situations and it's worth seeing them side by
side on the same images:

  detect_bounds=True  finds the organism with a text-prompted detector first,
                      then segments inside the box it found. For a photograph
                      where the organism's location is unknown.
  detect_bounds=False skips detection entirely and prompts SAM2 geometrically
                      -- positive point at the centre, negatives at the corners.
                      For an image that IS already a crop around one organism,
                      where detection would only re-find what the crop isolated.

On pre-cut crops the second should do at least as well as the first and cost
far less, since Grounding DINO is never loaded. On uncropped images the first
should win outright.

Needs a GPU to be anything but slow, and downloads model weights on first run.

Run from the repo root:
    python scripts/simple_tests/segmentation/groundedsam_test.py
"""

import logging
import time
from pathlib import Path

import cv2

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.segmentation.groundedsam import groundedsam2, sam2
from critterframe.visualization.panels import PanelFiles

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")
TEXT_PROMPT = "moth."

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
models = {
    "points  ": sam2(),
    "detected": groundedsam2(text_prompt=TEXT_PROMPT),
}

for label, model in models.items():
    print(f"\n== {label.strip()} ==")
    print("identity (this is what goes into the recipe hash):", model.identity())

    for path in sorted(CROP_DIR.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue

        segment = Segment(image, occurrence_id=f"{path.stem}_{label.strip()}",
                          project_path=PROJECT_PATH, panel_sink=PanelFiles(PROJECT_PATH))
        started = time.perf_counter()
        try:
            result, info = cf.segment(model)(segment)
            elapsed = time.perf_counter() - started
            print(f"  {path.name}: score {info['score']:.3f}  "
                  f"covers {info['area_fraction']:.1%} of the frame  "
                  f"{elapsed:.2f}s"
                  f"{'  RETRIED without centre point' if info.get('retried') else ''}")
        except Exception as exc:
            print(f"  {path.name}: FAILED -- {exc}")

print(f"\npanels in {PROJECT_PATH}/visualizations/segment/")
