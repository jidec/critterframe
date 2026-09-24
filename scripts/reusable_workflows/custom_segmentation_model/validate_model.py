"""
Step 4 of 4 of training a custom segmenter: score it against reference masks it
never trained on, beside the segmentation the project already has.

validate_masks(steps=...) runs the model live on every occurrence with a
reference mask and persists nothing. The scores are then split by the dataset's
own train/val/test labels: only TEST is an honest estimate (val chose the best
epoch), and a large train-test gap means overfitting. The current canonical
masks are scored on the same test occurrences, so the number to beat is right
beside the new one.

Writes a worst-first diff grid per comparison under visualizations/pipeline/
(white = agree, yellow = only predicted, red = only reference).

Before: register_model.py. If the model wins, use it in the pipeline:

    cf.run_segments(PROJECT_PATH, from_part=FROM_PART, shared_steps=TRANSFORMS,
                    outputs={PART: [cf.segment(segmentation.load_registered(PROJECT_PATH, MODEL_NAME))]})
"""

import logging

import pandas as pd

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PART = "organism"
DATASET_DIR = f"{PROJECT_PATH}/training/{PART}_segmenter"
MODEL_NAME = f"{PART}_segmenter_v1"

# The same framing prepare_training_data.py exported with.
FROM_PART = None
TRANSFORMS = []

# validate_masks(steps=) starts every occurrence from the bare image; it can't
# start from a stored upstream mask. So a part model trained with FROM_PART set
# needs the upstream mask recreated first, by the steps that made it, e.g.
#   UPSTREAM_STEPS = [cf.segment(cf.groundedsam2(text_prompt="insect."))]
UPSTREAM_STEPS = []

if FROM_PART is not None and not UPSTREAM_STEPS:
    raise ValueError(f"FROM_PART={FROM_PART!r} needs UPSTREAM_STEPS to recreate its mask")

model = segmentation.load_registered(PROJECT_PATH, MODEL_NAME)

candidate = cf.validate_masks(
    PROJECT_PATH, part=PART, label=MODEL_NAME,
    steps=[*UPSTREAM_STEPS, *TRANSFORMS, cf.segment(model)],
)

split_of = pd.read_csv(f"{DATASET_DIR}/manifest.csv",
                       dtype={"occurrence_id": str}).set_index("occurrence_id")["split"]
candidate = candidate.join(split_of, how="inner")
logging.info("%s IoU by split:\n%s", MODEL_NAME,
             candidate.groupby("split")["iou"].describe()[["count", "mean", "50%", "min"]].to_string())

# Baseline: the stored canonical masks, on the same held-out occurrences.
current = cf.validate_masks(PROJECT_PATH, part=PART, label="current_canonical")
test_ids = candidate.index[candidate["split"] == "test"]
baseline = current.loc[current.index.intersection(test_ids), "iou"]
logging.info("test IoU -- %s: %.3f (n=%d)   current masks: %.3f (n=%d)",
             MODEL_NAME, candidate.loc[test_ids, "iou"].mean(), len(test_ids),
             baseline.mean(), len(baseline))
