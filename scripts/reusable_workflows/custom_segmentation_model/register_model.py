"""
Step 3 of 4 of training a custom segmenter: record the checkpoint in the
project's model registry.

Registering fingerprints the weights and records what trained them (the
dataset.json from prepare_training_data.py, encoder, size). The fingerprint,
not the file name, is what reaches the recipe hash of every mask the model
makes: re-registering MODEL_NAME over new weights makes those masks, and every
metric measured from them, count as stale and be redone on the next run.

Before: train_model.py. Next: validate_model.py.
"""

import logging
from pathlib import Path

from critterframe.extensions.smp_segmenter import segmentation, training

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PART = "organism"
DATASET_DIR = f"{PROJECT_PATH}/training/{PART}_segmenter"
MODEL_NAME = f"{PART}_segmenter_v1"

# The same values train_model.py trained with.
ENCODER_NAME = segmentation.DEFAULT_ENCODER
SIZE = segmentation.DEFAULT_SIZE

# None registers the newest checkpoint train_model.py wrote; name one to pick
# an earlier training run instead.
CHECKPOINT = None

if CHECKPOINT is None:
    candidates = sorted(Path(DATASET_DIR).glob("smp_segmenter_*.pt"),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no smp_segmenter_*.pt in {DATASET_DIR} -- run train_model.py")
    CHECKPOINT = candidates[-1].resolve()

model = training.register_trained(
    PROJECT_PATH, MODEL_NAME, CHECKPOINT, DATASET_DIR,
    encoder_name=ENCODER_NAME, size=SIZE,
    notes=f"UNet++ for part '{PART}', trained on reference masks",
)
logging.info("registered %r", model)
