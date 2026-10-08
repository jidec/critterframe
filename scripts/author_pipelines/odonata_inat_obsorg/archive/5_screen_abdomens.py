"""
Step 5 of 6: screen a random sample of finished abdomen segments.

Each is given one key: good, or why it should not reach an export. Unlike a
reference mask this label exists for every input, so it is what the abdomen
filters (step 6) are calibrated against.

A person at a window. Resumes: rerun and only unlabelled abdomens are shown.

Before: 4_gate_and_segment_parts.py. Next: 6_filter_and_export.py.
"""

import logging

import critterframe as cf
from critterframe.project.subsets import select_ids

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

abdomens = set(cf.completed_ids(PROJECT_PATH, "body_parts", part="abdomen"))
gated = set(select_ids(PROJECT_PATH, subset="organism_gate_pass"))

cf.grow_subset(PROJECT_PATH, "part_screening_calibrate", target_size=100,
               candidate_ids=sorted(abdomens & gated),
               candidate_note="abdomens from body_parts that pass organism_gate_pass")

# in the organism crop the abdomen segmenter saw: the shared_steps
# 4_gate_and_segment_parts.py's body_parts run segments with
cf.run_metrics(
    PROJECT_PATH, part="abdomen",
    subset="part_screening_calibrate", from_part="organism",
    transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    metrics=[cf.exclusive_label_annotation(
        ["good", "input_invalid", "wrong_region", "incomplete", "overflow"],
        name="abdomen_quality",     # run_name defaults to "abdomen_quality"
        note="good: the mask is the abdomen. "
             "input_invalid: no correct abdomen mask exists in this image. "
             "wrong_region: the mask is on something else. "
             "incomplete: some of the abdomen is missing from the mask. "
             "overflow: the mask spills past the abdomen.")],
    visualize=False,
)
