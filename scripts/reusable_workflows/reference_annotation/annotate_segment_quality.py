"""
Screen finished part segments by hand, in two separate random samples.

Each segment is shown in the frame it was segmented in and given one key: good,
or the reason it should not reach an export (no valid input, wrong region,
incomplete, overflow). Unlike a reference mask, this label exists for every
input, so it is what says how much of an export is garbage.

Two samples, because thresholds chosen and scored on the same labels describe
those labels: validation/audit_filters.py calibrates on one and audits on the
other. Both are grown rather than redrawn, and each is drawn from outside the
other, so raising either size later never moves a segment across the line.

The labels describe the masks: resegmenting a part voids its labels, and the
next run of this script asks again.

Interruptible: rerun it and only unlabelled segments are shown.

Before: a canonical segmentation of every part in PARTS.
Next: validation/audit_filters.py.
"""

import logging

import critterframe as cf
from critterframe.project.subsets import select_ids
from critterframe.records.masks import occurrence_ids_with_mask

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PARTS = ["head", "thorax", "abdomen"]

# How the parts were segmented, so each panel is the frame the segmenter saw.
# For a part segmented straight from the image, use FROM_PART = None and no
# transforms.
FROM_PART = "organism"
TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient()]

CALIBRATE_SUBSET = "segment_quality_calibrate"
AUDIT_SUBSET = "segment_quality_audit"
CALIBRATE_SIZE = 300
AUDIT_SIZE = 300

QUALITY_RUN = "segment_quality"


def existing(subset):
    try:
        return set(select_ids(PROJECT_PATH, subset=subset))
    except KeyError:
        return set()


# The population that reaches an export: occurrences with a mask for every part.
pool = set.intersection(*(occurrence_ids_with_mask(PROJECT_PATH, part=part)
                          for part in PARTS))

cf.grow_subset(PROJECT_PATH, AUDIT_SUBSET, target_size=AUDIT_SIZE,
               candidate_ids=sorted(pool - existing(CALIBRATE_SUBSET)))
cf.grow_subset(PROJECT_PATH, CALIBRATE_SUBSET, target_size=CALIBRATE_SIZE,
               candidate_ids=sorted(pool - existing(AUDIT_SUBSET)))

# The vocabulary is yours. Decide it before screening: the labels are part of
# the recipe, so adding one later means screening again. Whatever it is, keep
# one label that means "fine"; validation/audit_filters.py names it as
# good_labels and treats every other label as bad.
segment_quality = cf.exclusive_label_annotation(
    ["good", "input_invalid", "wrong_region", "incomplete", "overflow"],
    name=QUALITY_RUN)

for subset in (CALIBRATE_SUBSET, AUDIT_SUBSET):
    cf.run_metrics(PROJECT_PATH, run_name=QUALITY_RUN, parts=PARTS, subset=subset,
                   from_part=FROM_PART, transforms=TRANSFORMS,
                   metrics=[segment_quality], visualize=False)

labels = cf.load_metrics(PROJECT_PATH, run_names=[QUALITY_RUN])
logging.info("labels so far:\n%s",
             labels.groupby("part")["value"].value_counts().to_string())
