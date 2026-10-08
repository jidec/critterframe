"""
Step 6 of 8: screen a random sample of finished head, thorax and abdomen
segments.

Each is given one key: perfect, good or poor. The label exists for every
segment, so it is what each part's filters (step 7) are calibrated against.

A person at a window. Resumes: rerun and only unlabelled segments are shown.

Before: 5_measure_part_qc_and_colors.py. Next: 7_filter_and_export.py.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

quality_label = cf.exclusive_label_annotation(
    labels=["perfect", "good", "poor"],
    note="These are judgements about segment shape, NOT illumination.\n"
         "perfect: the part entire, not curved, not covered by wing, and with <5% background bleed\n"
         "good: representative of color pattern of part but doesn't have to be entire part and can be curved\n"
         "poor: not representative of color pattern of part, not of part at all, has >5% background bleed\n",
    show_original=True)

# each part is shown as the colour traits see it: islands removed and the edge eroded
# head
cf.grow_subset(PROJECT_PATH, name="head_annotations", target_size=200)
cf.run_metrics(PROJECT_PATH, subset="head_annotations", part="head",
               transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                           cf.orient(), cf.erode(fraction=0.15)],
               metrics=[quality_label])
# thorax
cf.grow_subset(PROJECT_PATH, name="thorax_annotations", target_size=200)
cf.run_metrics(PROJECT_PATH, subset="thorax_annotations", part="thorax",
               transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                           cf.orient(), cf.erode(fraction=0.15)],
               metrics=[quality_label])
# abdomen
cf.grow_subset(PROJECT_PATH, name="abdomen_annotations", target_size=200)
cf.run_metrics(PROJECT_PATH, subset="abdomen_annotations", part="abdomen",
               transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                           cf.orient(axis_strategy="longer"), cf.erode(fraction=0.15)],
               metrics=[quality_label])
