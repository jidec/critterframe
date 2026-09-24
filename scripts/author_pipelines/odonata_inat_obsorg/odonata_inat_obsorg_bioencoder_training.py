"""
Trains a BioEncoder embedding model on odonata_inat_obsorg.py's organism
segments, with species as the metric-learning label, and registers the weights
so odonata_inat_obsorg.py's embedding step can load them.

Run directly (`python -m scripts.odonata_inat_obsorg.odonata_inat_obsorg_bioencoder_training`).
Also IMPORTABLE for EMBEDDING_TRANSFORMS and the model/run constants -- the
embedding metric must frame each organism exactly as this dataset export did --
so everything that does work is guarded by `if __name__ == "__main__":`.

Critterframe does the dataset side and BioEncoder only trains:

    critterframe                          BioEncoder
    ------------                          ----------
    choose + cap species        (step 1)
    split, grouped by observer  (step 2)
    export train/<species>/...  (step 3)
                                          configure   (step 5)
                                          train       (step 5, stage 1)
                                          swa         (step 5)
    register the swa weights    (step 6)

`bioencoder_split_dataset` is deliberately NOT used. It splits at random
within each class, so one observer's repeat photos of one animal land on both
sides and the validation score is inflated; `split_ids(group_col=)` is the
leakage guard it doesn't have. export_training_data(class_by=) already writes
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
from critterframe.project.subsets import select_occurrences
from critterframe.records.masks import occurrence_ids_with_mask
from critterframe.records.occurrences import ID_COL
from critterframe.selectionhelpers import cap_per_group
from critterframe.training.datasets import export_training_data
from critterframe.training.splits import split_ids

logger = logging.getLogger(__name__)

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
# The SAME chain odonata_inat_obsorg.py's embedding metric must use: a model
# trained on background-removed, cropped, oriented segments and run on
# anything else embeds badly with no error to say so. Import it from here.
EMBEDDING_TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient()]

LABEL_COL = "species"      # blank for genus-level identifications, which are left out
GROUP_COL = "recordedBy"   # one observer's repeat photos must not straddle train/val

# Species with fewer examples have too few positive pairs to learn from; more
# than the cap lets a few common species dominate every batch. BioEncoder's own
# split_dataset balances with max_ratio -- this is the same idea, applied once
# here since that step is skipped. 20 is also split_dataset's min_per_class.
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

    # Step 1: candidate occurrences -- an organism mask to cut the segment
    # from, a species to label it with, at most MAX_PER_CLASS per species,
    # and only species that still have MIN_PER_CLASS after that.
    occurrences = select_occurrences(PROJECT_PATH, columns=[LABEL_COL, GROUP_COL])
    occurrences = occurrences[
        occurrences[ID_COL].isin(occurrence_ids_with_mask(PROJECT_PATH, PART))
        & occurrences[LABEL_COL].notna()
        & (occurrences[LABEL_COL].astype(str).str.strip() != "")
    ]
    occurrences = cap_per_group(occurrences, LABEL_COL, MAX_PER_CLASS, seed=SEED)
    counts = occurrences[LABEL_COL].value_counts()
    occurrences = occurrences[occurrences[LABEL_COL].isin(counts[counts >= MIN_PER_CLASS].index)]
    logger.info("%d occurrences across %d species with >= %d examples",
                len(occurrences), occurrences[LABEL_COL].nunique(), MIN_PER_CLASS)

    # Step 2: split, stratified by species and grouped by observer.
    splits = split_ids(PROJECT_PATH, occurrence_ids=occurrences[ID_COL].tolist(),
                       fractions=FRACTIONS, stratify_col=LABEL_COL,
                       group_col=GROUP_COL, seed=SEED)

    # BioEncoder reads train/ and val/ as two separate ImageFolders, and an
    # ImageFolder numbers classes by its OWN sorted folder list -- a species
    # missing from val shifts every label after it, silently. Grouping by
    # observer can put all of a species on one side, so keep only species
    # present in both, and drop the rest from every split (test included)
    # BEFORE exporting, so dataset.json describes what was written.
    label_of = dict(zip(occurrences[ID_COL], occurrences[LABEL_COL]))
    shared = ({label_of[i] for i in splits["train"]}
              & {label_of[i] for i in splits["val"]})
    dropped = occurrences[LABEL_COL].nunique() - len(shared)
    if dropped:
        logger.warning("dropping %d species not present in both train and val", dropped)
    splits = {name: [i for i in ids if label_of[i] in shared]
              for name, ids in splits.items()}

    # Step 3: write BioEncoder's layout directly. Cleared first: the export
    # leaves unrelated existing files alone, and a species or image left over
    # from an earlier export would be trained on as though it were selected.
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    export_training_data(PROJECT_PATH, DATA_DIR, splits=splits, part=PART,
                         transforms=EMBEDDING_TRANSFORMS, class_by=LABEL_COL,
                         metadata=[GROUP_COL])

    # An occurrence can still fail to export (a transform failing on its
    # mask), so check the class lists once more on disk, where BioEncoder
    # will read them.
    train_classes = set(os.listdir(f"{DATA_DIR}/train"))
    val_classes = set(os.listdir(f"{DATA_DIR}/val"))
    if train_classes != val_classes:
        raise RuntimeError(
            f"train and val class folders differ after export "
            f"(train only: {sorted(train_classes - val_classes)}, "
            f"val only: {sorted(val_classes - train_classes)}) -- BioEncoder "
            "would misalign labels; see the export log for failed occurrences")

    # Step 4: the stage-1 config.
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(stage1_config(), handle, sort_keys=False)

    # Step 5: train. Imported here: bioencoder imports torch at import time,
    # and importing this module for EMBEDDING_TRANSFORMS shouldn't.
    # configure() sets root_dir/run_name for the calls after it (and writes
    # ~/.bioencoder.yaml for the CLI). train() needs CUDA. swa() averages the
    # last top_k_checkpoints epochs into weights/<run>/first/swa -- the file
    # BioEncoder's own inference loads, and the one registered below.
    import bioencoder

    bioencoder.configure(root_dir=BIOENCODER_ROOT, run_name=RUN_NAME, create=True)
    bioencoder.train(config_path=CONFIG_PATH, overwrite=True)
    bioencoder.swa(config_path=CONFIG_PATH)

    # Step 6: register. Fingerprints the swa weights, so retraining into the
    # same path changes every embedding recipe's hash; training_data points at
    # the export's dataset.json. `parameters` is what
    # extensions.bioencoder.training.load() is called with by
    # embedding.load_registered(PROJECT_PATH, MODEL_NAME), which is all
    # odonata_inat_obsorg.py needs to embed with this model (ImageNet mean/std,
    # which BioEncoder trained with, is load()'s default).
    cf.register_model(
        PROJECT_PATH, MODEL_NAME, path=CHECKPOINT, task="embedding",
        framework="torch", base_model=BACKBONE, training_data=DATA_DIR,
        parameters={"backbone": BACKBONE, "projection_head": True,
                    "input_size": [IMG_SIZE, IMG_SIZE], "normalize": True},
        notes=f"BioEncoder stage 1 (SupCon) + SWA; label={LABEL_COL}, "
              f"grouped by {GROUP_COL}; config {os.path.basename(CONFIG_PATH)}",
    )
