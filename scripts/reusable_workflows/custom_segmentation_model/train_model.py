"""
Step 2 of 4 of training a custom segmenter: train a UNet++ on the exported
dataset.

Needs the [torch] extra and, in practice, a GPU. Writes a checkpoint named
smp_segmenter_<encoder>_<date>[_n].pt into DATASET_DIR -- never over an earlier
one -- plus per-epoch prediction grids and loss/IoU curves under the project's
visualizations/pipeline/. Read those before registering: a validation IoU that
climbs and then falls is overfitting, and the grids show what it gets wrong.

Before: prepare_training_data.py. Next: register_model.py.
"""

import logging

from critterframe.extensions.smp_segmenter import segmentation, training

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PART = "organism"
DATASET_DIR = f"{PROJECT_PATH}/training/{PART}_segmenter"

# register_model.py must record these same two values, since loading the
# checkpoint later rebuilds the network from them.
ENCODER_NAME = segmentation.DEFAULT_ENCODER   # "efficientnet-b7"
SIZE = segmentation.DEFAULT_SIZE              # 352

NUM_EPOCHS = 30
BATCH_SIZE = 6       # lower this first on an out-of-memory error
VAL_SPLIT = "val"    # None if prepare_training_data.py made no val split

checkpoint = training.train(
    f"{DATASET_DIR}/manifest.csv", DATASET_DIR,
    encoder_name=ENCODER_NAME, size=SIZE,
    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, val_split=VAL_SPLIT,
    project_path=PROJECT_PATH,
)
logging.info("checkpoint: %s", checkpoint)
