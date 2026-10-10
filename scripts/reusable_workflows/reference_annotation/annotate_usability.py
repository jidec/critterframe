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

# The vocabulary is this project's own: edit the labels to the reasons worth
# telling apart here. run_name defaults to the metric's own name, "usability".
cf.run_metrics(PROJECT_PATH, subset=CANDIDATES_SUBSET,
               metrics=[cf.exclusive_label_annotation(
                       ["usable", "not_an_organism", "cut_off", "multiple_organisms", "wrong_life_stage",
                        "bad_angle", "dead", "broken_body", "obscured", "blurry", "overexposed",
                        "underexposed", "wrong_organism_for_project"],
                       name="usability",
                       requires_mask=False,     # a label about the image, so it can be asked before segmentation
                       note="usable: a single, complete organism\n"
                            "not_an_organism: nothing that should have been ingested\n"
                            "cut_off: an organism, but running off the frame edge\n"
                            "multiple_organisms: more than one in frame\n"
                            "wrong_life_stage: an organism, but not the stage this project studies\n"
                            "bad_angle: photographed from an angle that can't be measured reliably\n"
                            "dead: a dead specimen\n"
                            "broken_body: missing or damaged body parts\n"
                            "obscured: one organism, partly hidden behind debris, vegetation or another organism\n"
                            "blurry: too out of focus to trust\n"
                            "overexposed: too bright to trust\n"
                            "underexposed: too dark to trust\n"
                            "wrong_organism_for_project: segments fine, but isn't this project's subject\n"
                            "With more than one reason, give the one that also explains why the "
                            "segmentation can't be trusted: cut off and blurry is cut_off.")])

# Freeze the usable ones as a subset for the mask pass. Redefined on every run,
# so screening more of the sample reaches it without a separate step.
usable = cf.ids_matching(PROJECT_PATH, "usability", {"usability": "usable"})
cf.define_subset(PROJECT_PATH, USABLE_SUBSET, occurrence_ids=usable,
                 note=f"screened usable from {CANDIDATES_SUBSET}")

labels = cf.load_metrics(PROJECT_PATH, metric_names=["usability"])
logging.info("labels so far:\n%s", labels["value"].value_counts().to_string())
