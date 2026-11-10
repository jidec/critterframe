"""
Trains a BioEncoder embedding model on this pipeline's organism segments, with
species as the metric-learning label, and registers the weights so an embedding
step can load them.

Runs after carabidae_inat.py has segmented the organism. An embedding step
using this model must apply the same transforms as step 1's dataset export --
a model trained on background-removed, cropped, oriented segments and run on
anything else embeds badly with no error to say so.

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

Every organism mask is trained on, with no quality gate: the
quality_annotations_test labels in carabidae_inat.py are a 100-occurrence
sample, not yet a filter. A badly segmented beetle teaches the model a little
background; gate the dataset with `subset=` once a filter has been calibrated.

Only stage 1 (supervised-contrastive, projection head) is trained: that stage
produces the embedding space. Stage 2 only fits a classifier head on a frozen
encoder, which an embedding metric never uses.
"""

import logging
import os
import shutil

import critterframe as cf
from critterframe.extensions.bioencoder.training import prepare_dataset, register_trained

PROJECT_PATH = "D:/cf_projects/carabidae_inat"

# BioEncoder's root_dir
BIOENCODER_ROOT = f"{PROJECT_PATH}/training/bioencoder"

# Bump this for a new model rather than retraining over an old one:
# bioencoder.train(overwrite=True) deletes weights/<RUN_NAME>/first, which
# would pull the checkpoint out from under a model already registered from it.
RUN_NAME = "carabidae_v1"
MODEL_NAME = f"bioencoder_{RUN_NAME}"

DATA_DIR = f"{BIOENCODER_ROOT}/data/{RUN_NAME}"
CONFIG_PATH = f"{BIOENCODER_ROOT}/configs/{RUN_NAME}_stage1.yml"
CHECKPOINT = f"{BIOENCODER_ROOT}/weights/{RUN_NAME}/first/swa"


def stage1_config():
    """BioEncoder's train_stage1.yml, as a dict -- the package ships no template."""
    return {
        "model": {
            # timm_resnet18: the larger backbones failed to build or ran out of
            # GPU memory on this machine. Must match register_trained's backbone.
            "backbone": "timm_resnet18",
            # How many of the last epoch checkpoints bioencoder.swa() averages.
            "top_k_checkpoints": 3,
        },
        "train": {
            "n_epochs": 50,
            "amp": True,
            "ema": True,
            "ema_decay_per_epoch": 0.4,
            "target_metric": "precision_at_1",
            "stage": "first",
        },
        "dataloaders": {
            # The SupCon loss feeds two augmented crops per image, so memory
            # is ~2x the batch size.
            "train_batch_size": 24,
            # Validation drops its last incomplete batch, so this must stay
            # well under the size of the val split.
            "valid_batch_size": 24,
            "num_workers": 4,
        },
        "optimizer": {"name": "SGD", "params": {"lr": 0.003}},
        # T_max: the epoch count above.
        "scheduler": {"name": "CosineAnnealingLR",
                      "params": {"T_max": 50, "eta_min": 0.0003}},
        "criterion": {"name": "SupCon", "params": {"temperature": 0.1}},
        # Must match register_trained's input_size.
        "img_size": 256,
        "augmentations": {
            "sample_save": True,   # writes data/<run>/aug_sample/ -- look at it once
            "sample_n": 5,
            # Geometry only. No colour jitter / hue shifts: a carabid's
            # metallic sheen and elytral colour are exactly the signal the
            # embedding should keep. Both flips are fine for an oriented,
            # top-down beetle, which is near-symmetric left to right.
            "transforms": [
                {"RandomResizedCrop": {"height": 256, "width": 256,
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
    # organism mask and a species, any species missing from train or val
    # dropped from every split. Cleared first: prepare_dataset refuses a
    # directory that already holds images, since one left over from an
    # earlier export would be trained on.
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    prepare_dataset(
        PROJECT_PATH,
        DATA_DIR,
        label_col="species",      # blank for genus-level identifications, which are left out
        group_col="recordedBy",   # one observer's repeat photos must not straddle train/val
        part="organism",
        # The SAME chain the embedding step must use.
        transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
        # Species with fewer examples have too few positive pairs to learn
        # from; more than the cap lets a few common species dominate every
        # batch. The ingest already caps at 500 per scientificName, which is
        # far more than one species should contribute to an embedding's batches.
        min_per_label=20,
        max_per_label=100,
        fractions={"train": 0.80, "val": 0.20},   # test: never seen by BioEncoder
        seed=0,
    )

    # Step 2: the stage-1 config.
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(stage1_config(), handle, sort_keys=False)

    # Step 3: train. Imported here: bioencoder imports torch at import time.
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
    # the model with (ImageNet mean/std, which BioEncoder trained with, is
    # load()'s default). They are not in the model's identity: change one
    # under a new MODEL_NAME.
    register_trained(
        PROJECT_PATH,
        MODEL_NAME,
        checkpoint=CHECKPOINT,
        dataset_dir=DATA_DIR,
        backbone="timm_resnet18",    # the config's model.backbone
        input_size=(256, 256),       # the config's img_size
        projection_head=True,
        notes="BioEncoder stage 1 (SupCon) + SWA; label=species, grouped by recordedBy; "
              f"config {os.path.basename(CONFIG_PATH)}",
    )
