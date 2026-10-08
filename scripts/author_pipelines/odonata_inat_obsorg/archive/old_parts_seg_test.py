import logging

import critterframe as cf
from critterframe.metrics.outliers import outlier

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg_old"

from critterframe.extensions.smp_segmenter import segmentation
cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "head":    [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"))],
        "thorax":  [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"))],
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
    },
    visualize_every=100,
    force=True
)