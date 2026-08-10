"""
Dimension metrics over a folder of test crops, with a debug panel each.

Run these on ORIENTED masks or the numbers are meaningless -- body_length and
max_width measure image axes, and only orientation makes an image axis
correspond to the organism's own. The chain below is the real one:
remove_appendages, then orient, then measure.

max_width's panel is worth a look specifically. It draws the per-row width
profile beside the mask, and the maximum is an extremum, so one noisy row can
define it: a sustained peak is a real measurement, a lone spike is noise.

Run from the repo root:
    python scripts/simple_tests/metrics/dimensions_test.py
"""

import logging
from pathlib import Path

import cv2

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.segmentation.groundedsam import sam2
from critterframe.visualization.panels import PanelFiles

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
model = sam2()

transforms = [cf.remove_appendages(), cf.orient()]
metrics = [cf.body_length(), cf.max_width(), cf.mask_area(), cf.bounding_box()]

for path in sorted(CROP_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))
    if image is None:
        continue

    mask, _score, _info = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    segment = Segment(image, mask=mask, occurrence_id=path.stem,
                      project_path=PROJECT_PATH, panel_sink=PanelFiles(PROJECT_PATH))

    unreliable = False
    for transform in transforms:
        segment, info = transform(segment)
        unreliable = unreliable or info.get("unreliable", False) or \
            info.get("degenerate", False)

    values = {metric.metric_name: metric(segment) for metric in metrics}
    flag = "  <- orientation unreliable, treat with suspicion" if unreliable else ""
    print(f"{path.name}: length {values['body_length']:4d}px  "
          f"width {values['max_width']:4d}px  "
          f"area {values['mask_area']:6d}px  "
          f"box {values['bounding_box']}{flag}")

print(f"\npanels in {PROJECT_PATH}/visualizations/ (one subfolder per metric)")
