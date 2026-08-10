"""
Orientation over a folder of test crops, with a debug panel per crop.

The panel draws BOTH principal axes on the original -- the chosen one green,
the rejected one blue -- next to the oriented result. Seeing both is what makes
this checkable: the question isn't whether the mask rotated, it's whether the
asymmetry test picked the body over the wingspan, and a single drawn axis
wouldn't tell you.

Watch the eigenvalue ratio too. Near 1 means the mask is close to isotropic,
the axis directions are numerically unstable, and the result is flagged
unreliable rather than silently wrong.

Run from the repo root:
    python scripts/simple_tests/transforms/orient_test.py
"""

import logging
from pathlib import Path

import cv2

from critterframe.recipes import Segment
from critterframe.segmentation.groundedsam import sam2
from critterframe.transforms.orient import _orient, compute_orientation
from critterframe.visualization.panels import PanelFiles

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
model = sam2()

for path in sorted(CROP_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))
    if image is None:
        continue

    mask, _score, _info = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rotation, cx, cy, diagnostics = compute_orientation(mask)

    segment = Segment(image, mask=mask, occurrence_id=path.stem,
                      project_path=PROJECT_PATH, panel_sink=PanelFiles(PROJECT_PATH))
    oriented, info = _orient(segment)

    flag = "  UNRELIABLE" if info["unreliable"] else ""
    print(f"{path.name}: rot {rotation:+7.1f}deg  pc{info['chosen_pc']}  "
          f"skews {info['skew_pc0']:+.2f}/{info['skew_pc1']:+.2f}  "
          f"ratio {info['eigval_ratio']:.2f}  "
          f"longer_axis={info['chose_longer_axis']}  "
          f"{segment.shape}->{oriented.shape}{flag}")

print(f"\npanels in {PROJECT_PATH}/visualizations/orient/")
print("the canvas grows because rotating an upright body out of a wide frame "
      "needs room -- clipping the head or tail off would corrupt every length "
      "measured afterward")
