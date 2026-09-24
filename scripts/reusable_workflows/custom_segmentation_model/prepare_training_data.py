"""
Step 1 of 4 of training a custom segmenter: export the project's reference
masks as an image+mask dataset.

For a part the bundled GroundedSAM2 can't segment well on its own: a body part,
an organism against clutter, anything where hand-corrected masks exist
(reference_annotation/) and a small trained model can learn from them.

Writes DATASET_DIR/{train,val,test}/{images,masks}/<occurrence_id>.png plus
manifest.csv and dataset.json. dataset.json's data_hash is what the registered
model later points at, so the training data behind every mask it makes stays
on record.

Next: train_model.py.
"""

import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import training
from critterframe.records.masks import occurrence_ids_with_mask

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PART = "organism"
DATASET_DIR = f"{PROJECT_PATH}/training/{PART}_segmenter"
TRAINING_SUBSET = f"{PART}_segmenter_training"

# The frame the model sees. For a body part refined out of the organism mask,
# set FROM_PART = "organism" and crop to it, e.g.
#   TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient()]
# The inference run must use these exact transforms (as shared_steps) and this
# FROM_PART -- a model trained on one framing and run on another fails with no
# error, only bad masks.
FROM_PART = None
TRANSFORMS = []

# Leakage guard: occurrences sharing a value here (one specimen photographed
# several times, one observer's series) land on the same side of the split.
GROUP_COL = None   # e.g. "recordedBy"

# val picks the best epoch; test is held out for validate_model.py, which is
# the only honest score. A reference set too small for three splits can drop
# "val" here and train with VAL_SPLIT = None in train_model.py.
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Every occurrence with a reference mask for this part, frozen as a subset so
# the split is reproducible and visible in subsets.toml.
reference_ids = sorted(occurrence_ids_with_mask(PROJECT_PATH, PART, reference=True))
cf.define_subset(PROJECT_PATH, TRAINING_SUBSET, occurrence_ids=reference_ids,
                 note=f"every reference {PART} mask, for {DATASET_DIR}")
logging.info("%d reference %s masks to train from", len(reference_ids), PART)

# reference=True is prepare_dataset's default: training on the canonical masks
# would teach the new model the old one's mistakes.
manifest = training.prepare_dataset(
    PROJECT_PATH, DATASET_DIR, part=PART, from_part=FROM_PART,
    transforms=TRANSFORMS, subset=TRAINING_SUBSET, fractions=FRACTIONS,
    group_col=GROUP_COL,
)
logging.info("exported:\n%s", manifest["split"].value_counts().to_string())
