import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# mask_threshold is a logit where 1.0 keeps only the pixels each segmenter is surest of so a mask is tighter than at the neutral 0.0.
cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts_tighter",
    from_part="organism",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "head": [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"), mask_threshold=1.0)],
        "thorax": [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"), mask_threshold=1.0)],
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"), mask_threshold=1.0)],
    },
)
