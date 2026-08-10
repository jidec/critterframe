"""
Automated QC metrics over a folder of test crops, with a debug panel each.

Print the scores next to each other and see whether they separate the crops you
already know are bad from the ones you know are good. If they don't, the metric
is the wrong instrument for this data and no threshold will save it -- which is
worth finding out here rather than after calibrating one.

The WARN constants these are compared against are eyeballed defaults, not
derived. Treat the flags below as a sanity check, and calibrate real thresholds
against human labels with scripts/validation_pipeline.py.

Run from the repo root:
    python scripts/simple_tests/metrics/quality_test.py
"""

import logging
from pathlib import Path

import cv2

import critterframe as cf
from critterframe.metrics.quality import (
    ASYMMETRY_WARN_SCORE,
    BLUR_WARN_VARIANCE,
    EDGE_WARN_FRACTION,
)
from critterframe.recipes import Segment
from critterframe.segmentation.groundedsam import sam2
from critterframe.visualization.panels import PanelFiles

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
model = sam2()

print(f"defaults: blur < {BLUR_WARN_VARIANCE} is blurry, "
      f"asymmetry > {ASYMMETRY_WARN_SCORE} is lopsided, "
      f"edge > {EDGE_WARN_FRACTION} is cut off\n")

for path in sorted(CROP_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))
    if image is None:
        continue

    mask, _score, _info = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    segment = Segment(image, mask=mask, occurrence_id=path.stem,
                      project_path=PROJECT_PATH, panel_sink=PanelFiles(PROJECT_PATH))

    # blur, edge fraction, and mask fraction are rotation-invariant; bilateral
    # asymmetry is not, so it needs the oriented mask -- on an unoriented one
    # the left/right split is arbitrary.
    blur = cf.blur_variance()(segment)
    edge = cf.edge_fraction()(segment)
    coverage = cf.mask_fraction()(segment)

    oriented, info = cf.orient()(segment)
    asymmetry = cf.bilateral_asymmetry()(oriented)

    flags = []
    if blur < BLUR_WARN_VARIANCE:
        flags.append("BLURRY")
    if asymmetry > ASYMMETRY_WARN_SCORE:
        flags.append("ASYMMETRIC")
    if edge > EDGE_WARN_FRACTION:
        flags.append("CUT OFF")
    if info["unreliable"]:
        flags.append("ORIENTATION UNRELIABLE")

    print(f"{path.name}: blur {blur:8.1f}  asymmetry {asymmetry:.3f}  "
          f"edge {edge:.3f}  coverage {coverage:.3f}  "
          f"{'  '.join(flags)}")

print(f"\npanels in {PROJECT_PATH}/visualizations/ (one subfolder per metric)")
