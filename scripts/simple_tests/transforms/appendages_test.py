"""
Appendage removal over a folder of test crops, with a debug panel per crop.

The panel is the point: original mask in grey, retained body in white, REMOVED
pixels in red. Look at whether what came off was legs and antennae rather than
a wing tip or a tapering abdomen -- a removed_fraction number alone can't tell
you which.

Needs a folder of test crops and a segmenter. Panels are written to
PROJECT_PATH/visualizations/remove_appendages/.

Run from the repo root:
    python scripts/simple_tests/transforms/appendages_test.py
"""

import logging
import sys
from pathlib import Path

from critterframe.transforms.appendages import _remove_appendages

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _crop_folder import segmented_crops          # noqa: E402

PROJECT_PATH = "projects/scratch"          # only used for writing visualizations
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
for path, segment, score in segmented_crops(CROP_DIR, PROJECT_PATH):
    cleaned, info = _remove_appendages(segment)

    flag = "  DEGENERATE" if info["degenerate"] else ""
    print(f"{path.name}: sam {score:.3f}  radius {info['radius']}px  "
          f"removed {info['removed_fraction']:6.1%}  "
          f"{info['area_before']}->{info['area_after']}px  "
          f"components {info['n_components']}{flag}")

print(f"\npanels in {PROJECT_PATH}/visualizations/remove_appendages/")
