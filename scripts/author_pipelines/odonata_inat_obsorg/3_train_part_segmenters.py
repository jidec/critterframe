import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation, training
from critterframe.selection.subsets import define_subset

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
DATASET_DIR = f"{PROJECT_PATH}/training/body_parts"

# transforms used to create the training set (and intended to be applied BEFORE applying the segmenters later)
BODY_PART_TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")]

# newly drawn reference masks reach training without anyone remembering to redefine the subset by hand.
head_ids = set(cf.ids_with_mask(PROJECT_PATH, part="head", reference=True))

# remove ids with annotation mistakes
training_ids = sorted(head_ids - {"4177294498","2273301679","2563482116","3070675391","3747273588","3759334647",
                                  "3873762939","3873771580","4162072296","4171342892","4522442468","4889841733",
                                  "5840134655","3873762939","3873771580","4171342892","4889841733","5901775605",
                                  "5902514590"})
define_subset(PROJECT_PATH, "body_parts_training", occurrence_ids=training_ids)

# prepare training datasets
head_dataset = training.prepare_dataset(
    PROJECT_PATH, f"{DATASET_DIR}/head", part="head", from_part="organism",
    transforms=BODY_PART_TRANSFORMS, subset="body_parts_training",
    fractions={"train": 0.85, "test": 0.15},
)
thorax_dataset = training.prepare_dataset(
    PROJECT_PATH, f"{DATASET_DIR}/thorax", part="thorax", from_part="organism",
    transforms=BODY_PART_TRANSFORMS, subset="body_parts_training",
    fractions={"train": 0.85, "test": 0.15},
)
abdomen_dataset = training.prepare_dataset(
    PROJECT_PATH, f"{DATASET_DIR}/abdomen", part="abdomen", from_part="organism",
    transforms=BODY_PART_TRANSFORMS, subset="body_parts_training",
    fractions={"train": 0.85, "test": 0.15},
)

# train segmenters then register their provenance so they can be used in critterframe
ENCODER_NAME = segmentation.DEFAULT_ENCODER
TRAINING_SIZE = segmentation.DEFAULT_SIZE

# head
head_checkpoint = training.train(head_dataset, f"{DATASET_DIR}/head",
                                 encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
                                 val_split=None, project_path=PROJECT_PATH)
training.register_trained(
    PROJECT_PATH, "head_segmenter_v1", head_checkpoint,
    f"{DATASET_DIR}/head",
    encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
)

# thorax
thorax_checkpoint = training.train(thorax_dataset, f"{DATASET_DIR}/thorax",
                                   encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
                                   val_split=None, project_path=PROJECT_PATH)
training.register_trained(
    PROJECT_PATH, "thorax_segmenter_v1", thorax_checkpoint,
    f"{DATASET_DIR}/thorax",
    encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
)
# abdomen
abdomen_checkpoint = training.train(abdomen_dataset, f"{DATASET_DIR}/abdomen",
                                    encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
                                    val_split=None, project_path=PROJECT_PATH)
training.register_trained(
    PROJECT_PATH, "abdomen_segmenter_v1", abdomen_checkpoint,
    f"{DATASET_DIR}/abdomen",
    encoder_name=ENCODER_NAME, size=TRAINING_SIZE,
)
