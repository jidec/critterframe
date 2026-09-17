"""
Trains odonata_inat_obsorg.py's head/thorax/abdomen segmenters from this
project's own hand-drawn reference masks.

Run directly (`python -m scripts.odonata_inat_obsorg.odonata_inat_obsorg_part_segmenters_training_validation`)
to build the training set, export a dataset, and train+register a model per
part. Also IMPORTABLE for BODY_PART_TRANSFORMS -- what
odonata_inat_obsorg.py's part-segmentation step needs to frame each part's
crop the same way this script's dataset export did -- which is why
everything that actually does work below is guarded by
`if __name__ == "__main__":`: importing this module must not re-run a
training pass (or block on GPU work) every time the main pipeline starts.
"""

import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation, training
from critterframe.project.subsets import define_subset, select_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
DATASET_DIR = f"{PROJECT_PATH}/training/body_parts"

PARTS = ["head", "thorax", "abdomen"]
BODY_PART_TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")]

# Read ONCE here rather than each call site reading
# segmentation.DEFAULT_ENCODER/DEFAULT_SIZE separately -- training.train()
# and cf.register_model() below both need the SAME values, and pinning them
# to one pair of constants is what actually prevents them drifting apart if
# those module defaults ever change between one run and a later one.
# segmentation.load_registered() reads these back out of the registry at
# load time instead of re-reading whatever the module defaults happen to be
# by then.
ENCODER_NAME = segmentation.DEFAULT_ENCODER
TRAINING_SIZE = segmentation.DEFAULT_SIZE

# Step 0: combine this project's two reference-mask subsets into one
# training set. "old_but_valid_parts_reference_set" is a fixed, one-off
# batch of hand-drawn part masks made early in this project -- frozen on
# purpose, nothing grows it further. "mask_reference_set" is newer
# reference masks, grown over time (elsewhere, via
# critterframe.project.subsets.grow_subset) as more specimens get
# hand-corrected. Redefining "body_parts_training_combined" fresh on
# every run -- rather than growing it once and leaving it -- is what
# makes mask_reference_set's own growth actually reach training:
# define_subset(occurrence_ids=...) freezes whatever list it's given at
# call time (same mechanism grow_subset() itself uses), so without this
# a bigger mask_reference_set later wouldn't be reflected until someone
# remembered to redefine the training subset by hand.
old_ids = select_ids(PROJECT_PATH, subset="old_but_valid_parts_reference_set")
new_ids = select_ids(PROJECT_PATH, subset="mask_reference_set")
combined_ids = sorted((set(old_ids) | set(new_ids)) - {"4177294498"})
define_subset(PROJECT_PATH, "body_parts_training_combined", occurrence_ids=combined_ids)

# Step 1: export one image+mask dataset per part.
#
# training.prepare_dataset() (extensions/smp_segmenter/training.py) is a
# thin wrapper: it calls training.splits.split_ids() to divide the
# training subset's occurrences into train/test groups (no "val" split
# here -- see the val_split=None note on training.train() below), then
# training.datasets.export_training_data() to actually write files:
#   D:/cf_projects/odonata_inat_obsorg/training/body_parts/<part>/
#       train/images/<occurrence_id>.png   train/masks/<occurrence_id>.png
#       test/images/<occurrence_id>.png     test/masks/<occurrence_id>.png
#       manifest.csv                        dataset.json
# `reference=True` (prepare_dataset's default) reads masks from
# reference_masks.parquet rather than the canonical masks.parquet --
# training on the automated masks would just teach a new model to
# reproduce the old one's mistakes. `from_part="organism"` means each
# part's crop/orient/remove_background is applied to the ORGANISM mask's
# frame first (matching how odonata_inat_obsorg.py's from_part="organism"
# segmentation run frames things at inference), and the part's own
# (smaller) reference mask is then reprojected into that same frame.
# dataset.json's `data_hash` is what register_model() below points at,
# so retraining from a changed dataset is provably a different dataset.
datasets = {
    part: training.prepare_dataset(
        PROJECT_PATH, f"{DATASET_DIR}/{part}", part=part, from_part="organism",
        transforms=BODY_PART_TRANSFORMS, subset="body_parts_training_combined",
        fractions={"train": 0.75, "test": 0.25},
    )
    for part in PARTS
}

# Step 2: train one segmenter per part, then register the checkpoint.
for part in PARTS:
    # training.train() (extensions/smp_segmenter/training.py) is a real,
    # runnable training loop: an ImageNet-pretrained UnetPlusPlus encoder,
    # BCEWithLogitsLoss, Adam. val_split=None here because our reference
    # sets are small -- with no third split to validate against, train()
    # trains for every epoch with no early stopping and saves the FINAL
    # epoch's weights (rather than tracking a best-validation-loss epoch)
    # to a checkpoint file it names itself:
    #   D:/cf_projects/odonata_inat_obsorg/training/body_parts/<part>/
    #       smp_segmenter_<encoder>_<today>[_n].pt
    # The `_n` suffix means retraining twice in one day never silently
    # overwrites a checkpoint register_model() already fingerprinted below.
    # encoder_name/size are passed explicitly (ENCODER_NAME/TRAINING_SIZE,
    # defined above) rather than left to train()'s own defaults, so the
    # exact values used here are the same ones handed to register_model()
    # just below -- see that call's `parameters=` for why.
    checkpoint = training.train(datasets[part], f"{DATASET_DIR}/{part}",
                                encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
                                val_split=None,show=True)

    # cf.register_model() (records/models.py) is provenance, not
    # training: it never loads the checkpoint into a network, only
    # reads its bytes to compute a sha256 fingerprint (stored as
    # `fingerprint`/`fingerprint_method`) and records the checkpoint
    # path RELATIVE to the project directory (so a copied/moved project
    # still resolves it), alongside task/framework/base_model and a
    # pointer at the dataset.json this part's dataset export just wrote
    # (`training_data=`). That fingerprint -- not the name or the path --
    # is what RegisteredModel.identity() feeds into a segmentation
    # recipe's hash: retraining this same `f"{part}_segmenter_v1"` name
    # into a new checkpoint file moves the fingerprint, which moves the
    # recipe hash, which is what makes every mask (and everything
    # measured off it) downstream of this part correctly count as stale
    # and get redone on the next run_segments() pass. Registering under
    # a name that already exists replaces the record and logs loudly
    # when the fingerprint changed, exactly to make that consequence
    # visible.
    #
    # `parameters={"encoder_name": ..., "size": ...}` is what closes a
    # separate gap: SMPSegmenter needs both to load a checkpoint
    # correctly, and without recording them here, loading later would
    # have to assume DEFAULT_ENCODER/DEFAULT_SIZE in
    # extensions/smp_segmenter/segmentation.py hadn't changed since this
    # model was trained. segmentation.load_registered() reads them back
    # from exactly this field, so a later change to those module
    # defaults can never silently swap the architecture under this
    # checkpoint.
    cf.register_model(
        PROJECT_PATH, f"{part}_segmenter_v1", path=checkpoint,
        task="segment", framework="torch",
        base_model=ENCODER_NAME,
        training_data=f"{DATASET_DIR}/{part}",
        parameters={"encoder_name": ENCODER_NAME, "size": TRAINING_SIZE},
    )
