"""
How good are the project's measurements? Compare an automated metric against
the same quantity measured by hand.

A mask can score well on IoU and still give the wrong length -- a missing
abdomen tip is a few pixels of area and a real share of the body. So a trait
worth publishing gets its own check: click the two ends of each reference
specimen, then compare body_length against the clicked length.

compare_metrics reports, per pair: mean and median percent difference (a mean
far above the median means one bad reference click), bias (a steady offset is
correctable, noise is not), and correlation. It writes a predicted-vs-reference
scatter and a grid of the worst disagreements under visualizations/pipeline/.

Before: reference_annotation/ for the subset to click on.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
PART = "organism"
REFERENCE_SUBSET = "mask_reference_set"

PREDICTED_RUN = "body_dimensions"
REFERENCE_RUN = "length_by_hand"

# 1. The automated measurement. Skips everything already measured, so this is
#    free if the main pipeline already ran it under the same recipe.
cf.run_metrics(
    PROJECT_PATH, run_name=PREDICTED_RUN, part=PART,
    transforms=[cf.remove_appendages(), cf.orient()],
    metrics=[cf.body_length(), cf.max_width()],
)

# 2. The hand measurement: click head, then tail, on each reference specimen.
#    Interruptible -- a rerun shows only what's left. Esc skips a specimen.
cf.run_metrics(
    PROJECT_PATH, run_name=REFERENCE_RUN, part=PART, subset=REFERENCE_SUBSET,
    metrics=[cf.click_two_points(labels=("head", "tail"))],
)

# 3. Compare. The dict pairs differently-named metrics; a dict-valued metric
#    is addressed by "<metric>__<key>".
comparison = cf.compare_metrics(
    PROJECT_PATH, predicted_run=PREDICTED_RUN, reference_run=REFERENCE_RUN,
    metric_names={"body_length": "click_two_points__length_px"}, part=PART,
)
logging.info("\n%s", comparison.to_string(index=False))
