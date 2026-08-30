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
import sys
from pathlib import Path

import critterframe as cf
from critterframe.metrics.quality import (
    ASYMMETRY_WARN_SCORE,
    BLUR_WARN_VARIANCE,
    EDGE_WARN_FRACTION,
)

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _crop_folder import segmented_crops          # noqa: E402

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
print(f"defaults: blur < {BLUR_WARN_VARIANCE} is blurry, "
      f"asymmetry > {ASYMMETRY_WARN_SCORE} is lopsided, "
      f"edge > {EDGE_WARN_FRACTION} is cut off\n")

for path, segment, _score in segmented_crops(CROP_DIR, PROJECT_PATH):
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
