"""
Step 2 of 3 of building a reference set: screen the sample by hand.

Each occurrence is shown in a window and given one key: usable, or the reason
it isn't (cut off, multiple organisms, blurry, ...). No mask is needed, so this
can run before segmentation. The labels do two jobs: they decide which
occurrences get reference masks (a crop with no single boundary to draw only
makes the reference set lie), and they are what
validation/validate_filters.py calibrates QC cutoffs against.

Interruptible: rerun it and only unlabelled occurrences are shown.

Before: create_reference_sample.py. Next: annotate_reference_masks.py.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

CANDIDATES_SUBSET = "reference_candidates"
USABLE_SUBSET = "reference_candidates_usable"

# run_name defaults to the metric's own name, "usability_annotation".
cf.run_metrics(PROJECT_PATH, subset=CANDIDATES_SUBSET,
               metrics=[cf.usability_annotation()])

# Freeze the usable ones as a subset for the mask pass. Redefined on every run,
# so screening more of the sample reaches it without a separate step.
usable = cf.occurrences_matching(PROJECT_PATH, "usability_annotation",
                                 {"usability_annotation": "usable"})
cf.define_subset(PROJECT_PATH, USABLE_SUBSET, occurrence_ids=usable,
                 note=f"screened usable from {CANDIDATES_SUBSET}")

labels = cf.load_metrics(PROJECT_PATH, metric_names=["usability_annotation"])
logging.info("labels so far:\n%s", labels["value"].value_counts().to_string())
