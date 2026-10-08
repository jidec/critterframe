"""
Step 5 of 8: measure every head, thorax and abdomen: the QC scores the filters
are calibrated on, an embedding, and the colour traits that get exported.

Every segmented part is measured; nothing is filtered here.

Unattended, on a GPU for the embeddings.

Before: 4_segment_parts.py. Next: 6_annotate_parts_for_filters.py.
"""

import logging

import critterframe as cf
from critterframe.metrics.embedding import pretrained

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# qc scores: candidates for filtering out bad segments
cf.run_metrics(PROJECT_PATH, run_name="qc", parts=["head", "thorax", "abdomen"],
               metrics=[cf.mask_info(operations=["segment"]),
                        cf.mask_fraction(), cf.blur_variance(), cf.mask_area(), cf.n_islands(),
                        cf.elongation(), cf.jaggedness()])

# embeddings
cf.run_metrics(PROJECT_PATH, run_name="resnet18_embedding", parts=["head", "thorax", "abdomen"],
               transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                           cf.orient(axis_strategy="longer"), cf.crop_to_mask()],
               metrics=[cf.embedding(pretrained("resnet18", resize="pad"), name="resnet18_embedding")])

# every bin also needs high chroma, so browns, greys and pruinose whites land in "unmatched".
# light blue gets a lower chroma floor because pale blues are inherently low-chroma (sky blue ~26, brown ~24)
cf.run_metrics(PROJECT_PATH, run_name="color_thresholds", parts=["head", "thorax", "abdomen"],
               transforms=[cf.remove_background(), cf.remove_islands(), cf.erode(fraction=0.15)],
               metrics=[cf.threshold_fractions([
                            cf.color_threshold("red",        lch_h=(345, 50),  lch_c=(30, None)),
                            cf.color_threshold("orange",     lch_h=(50, 75),   lch_c=(30, None)),
                            cf.color_threshold("yellow",     lch_h=(75, 105),  lch_c=(30, None)),
                            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
                            cf.color_threshold("blue",       lch_h=(195, 305), lch_c=(30, None), lch_l=(None, 55)),
                            cf.color_threshold("light_blue", lch_h=(195, 305), lch_c=(20, None), lch_l=(55, None)),
                            cf.color_threshold("purple",     lch_h=(305, 345), lch_c=(30, None)),
                        ], unmatched=True, name="color_bins"),
                        # present at >= 5% of the part; n_colors_present; ranked_color_1/2
                        cf.color_presence(min_fraction=0.05, n_ranked_colors=2),
                    ])