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
import sys
from pathlib import Path

import critterframe as cf

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _crop_folder import segmented_crops          # noqa: E402

PROJECT_PATH = "projects/scratch"
CROP_DIR = Path("scripts/test_images/insect_crops")

# PROJECT_PATH needn't exist -- the panel sink creates what it writes into
transforms = [cf.remove_appendages(), cf.orient()]
metrics = [cf.body_length(), cf.max_width(), cf.mask_area(), cf.bounding_box()]

for path, segment, _score in segmented_crops(CROP_DIR, PROJECT_PATH):
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
