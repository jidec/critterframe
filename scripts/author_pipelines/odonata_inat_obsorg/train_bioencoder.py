"""
Trains a BioEncoder embedding model on this pipeline's organism segments, with
species as the metric-learning label, and registers the weights so an embedding
step can load them.

Outside the numbered sequence, and optional: 6_filter_and_export.py embeds with
an off-the-shelf resnet18 today.
Also IMPORTABLE for EMBEDDING_TRANSFORMS and the model/run constants -- the
embedding metric must frame each organism exactly as this dataset export did --
so everything that does work is guarded by `if __name__ == "__main__":`.

Critterframe does the dataset side and BioEncoder only trains:

    critterframe                          BioEncoder
    ------------                          ----------
    prepare_dataset             (step 1)
      choose + cap species
      split, grouped by observer
      export train/<species>/...
                                          configure   (step 3)
                                          train       (step 3, stage 1)
                                          swa         (step 3)
    register_trained            (step 4)

`bioencoder_split_dataset` is deliberately NOT used. It splits at random
within each class, so one observer's repeat photos of one animal land on both
sides and the validation score is inflated; the extension's prepare_dataset
splits with `group_col=`, the leakage guard it doesn't have, and writes
BioEncoder's own `<root>/data/<run>/{train,val}/<class>/` layout.

Only stage 1 (supervised-contrastive, projection head) is trained: that stage
produces the embedding space. Stage 2 only fits a classifier head on a frozen
encoder, which an embedding metric never uses.

BioEncoder is called through its Python API, not the `bioencoder_*` commands.
The CLI hands root_dir/run_name between processes via ~/.bioencoder.yaml and
swallows tracebacks inside subprocess output; in-process, a failure is an
ordinary exception.
"""

import logging
import os
import shutil

import critterframe as cf
from critterframe.extensions.bioencoder.training import prepare_dataset, register_trained

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# BioEncoder's root_dir. It lays everything out beneath it by run name:
#   data/<RUN_NAME>/{train,val,test}/<species>/<occurrence_id>.png  (written here)
#   weights/<RUN_NAME>/first/{epoch<N>, swa}                        (BioEncoder)
#   logs/, runs/ (tensorboard)                                      (BioEncoder)
BIOENCODER_ROOT = f"{PROJECT_PATH}/training/bioencoder"

# Bump this for a new model rather than retraining over an old one:
# bioencoder.train(overwrite=True) deletes weights/<RUN_NAME>/first, which
# would pull the checkpoint out from under a model already registered from it.
RUN_NAME = "odonata_v1"
MODEL_NAME = f"bioencoder_{RUN_NAME}"

DATA_DIR = f"{BIOENCODER_ROOT}/data/{RUN_NAME}"
CONFIG_PATH = f"{BIOENCODER_ROOT}/configs/{RUN_NAME}_stage1.yml"
CHECKPOINT = f"{BIOENCODER_ROOT}/weights/{RUN_NAME}/first/swa"

PART = "organism"
# The SAME chain the pipeline's embedding metric must use: a model
# trained on background-removed, cropped, oriented segments and run on
# anything else embeds badly with no error to say so. Import it from here.
EMBEDDING_TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient()]

LABEL_COL = "species"      # blank for genus-level identifications, which are left out
GROUP_COL = "recordedBy"   # one observer's repeat photos must not straddle train/val

# Species with fewer examples have too few positive pairs to learn from; more
# than the cap lets a few common species dominate every batch. BioEncoder's own
# split_dataset balances with max_ratio -- this is the same idea, applied once
# here since that step is skipped.
MIN_PER_CLASS = 20
MAX_PER_CLASS = 100
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}   # test: never seen by BioEncoder
SEED = 0

# timm_resnet18 rather than BioEncoder's example efficientnet_b5: the larger
# backbones failed to build or ran out of GPU memory on this machine. The SupCon
# loss feeds two augmented crops per image, so memory is ~2x batch_size.
BACKBONE = "timm_resnet18"
IMG_SIZE = 256
EPOCHS = 50
BATCH_SIZE = 24


def stage1_config():
    """BioEncoder's train_stage1.yml, as a dict -- the package ships no template."""
    return {
        "model": {
            "backbone": BACKBONE,
            # How many of the last epoch checkpoints bioencoder.swa() averages.
            "top_k_checkpoints": 3,
        },
        "train": {
            "n_epochs": EPOCHS,
            "amp": True,
            "ema": True,
            "ema_decay_per_epoch": 0.4,
            "target_metric": "precision_at_1",
            "stage": "first",
        },
        "dataloaders": {
            "train_batch_size": BATCH_SIZE,
            # Validation drops its last incomplete batch, so this must stay
            # well under the size of the val split.
            "valid_batch_size": BATCH_SIZE,
            "num_workers": 4,
        },
        "optimizer": {"name": "SGD", "params": {"lr": 0.003}},
        "scheduler": {"name": "CosineAnnealingLR",
                      "params": {"T_max": EPOCHS, "eta_min": 0.0003}},
        "criterion": {"name": "SupCon", "params": {"temperature": 0.1}},
        "img_size": IMG_SIZE,
        "augmentations": {
            "sample_save": True,   # writes data/<run>/aug_sample/ -- look at it once
            "sample_n": 5,
            # Geometry only. No colour jitter / hue shifts: colour is exactly
            # the signal an odonate embedding should keep, and augmenting it
            # away trains the model to ignore it.
            "transforms": [
                {"RandomResizedCrop": {"height": IMG_SIZE, "width": IMG_SIZE,
                                       "scale": [0.7, 1.0]}},
                {"HorizontalFlip": None},
                {"VerticalFlip": None},
                {"ShiftScaleRotate": {"p": 0.4}},
                {"MedianBlur": {"p": 0.3}},
            ],
        },
    }


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    # Step 1: the dataset, in BioEncoder's layout. Only occurrences with an
    # organism mask and a species, at most MAX_PER_CLASS per species, species
    # with fewer than MIN_PER_CLASS left out, split grouped by observer, and
    # any species missing from train or val dropped from every split. Cleared
    # first: prepare_dataset refuses a directory that already holds images,
    # since one left over from an earlier export would be trained on.
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    prepare_dataset(
        PROJECT_PATH,
        DATA_DIR,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        part=PART,
        transforms=EMBEDDING_TRANSFORMS,
        min_per_label=MIN_PER_CLASS,
        max_per_label=MAX_PER_CLASS,
        fractions=FRACTIONS,
        seed=SEED,
    )

    # Step 2: the stage-1 config.
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(stage1_config(), handle, sort_keys=False)

    # Step 3: train. Imported here: bioencoder imports torch at import time,
    # and importing this module for EMBEDDING_TRANSFORMS shouldn't.
    # configure() sets root_dir/run_name for the calls after it (and writes
    # ~/.bioencoder.yaml for the CLI). train() needs CUDA. swa() averages the
    # last top_k_checkpoints epochs into weights/<run>/first/swa -- the file
    # BioEncoder's own inference loads, and the one registered below.
    import bioencoder

    bioencoder.configure(root_dir=BIOENCODER_ROOT, run_name=RUN_NAME, create=True)
    bioencoder.train(config_path=CONFIG_PATH, overwrite=True)
    bioencoder.swa(config_path=CONFIG_PATH)

    # Step 4: register. Fingerprints the swa weights, so retraining into the
    # same path changes every embedding recipe's hash; training_data points at
    # the export's dataset.json. The backbone, input size and head recorded
    # here are what embedding.load_registered(PROJECT_PATH, MODEL_NAME) loads
    # the model with, which is all the pipeline needs to embed with it
    # (ImageNet mean/std, which BioEncoder trained with, is load()'s default).
    # They are not in the model's identity: change one under a new MODEL_NAME.
    register_trained(
        PROJECT_PATH,
        MODEL_NAME,
        checkpoint=CHECKPOINT,
        dataset_dir=DATA_DIR,
        backbone=BACKBONE,
        input_size=(IMG_SIZE, IMG_SIZE),
        projection_head=True,
        notes=f"BioEncoder stage 1 (SupCon) + SWA; label={LABEL_COL}, "
              f"grouped by {GROUP_COL}; config {os.path.basename(CONFIG_PATH)}",
    )
