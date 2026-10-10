"""
Step 2 of 8: label each image usable or not_usable, and correct reference part
masks on the usable ones.

Usability labels decide which images get reference masks: an image with no
single complete organism has no correct part mask to draw. The reference masks are what the part
segmenters (step 3) train on.

A person at a window. Each pass resumes: rerun and only what is unlabelled is
shown.

Before: 1_ingest_download_and_segment_organism.py. Next: 3_train_part_segmenters.py.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# annotate usability so we don't try to put any invalid images in the part segmenter training sets
cf.grow_subset(PROJECT_PATH, name="usability_annotation_set", target_size=150)
# the label describes the image, not its mask, so it is asked of every image and survives resegmenting.
# run_name defaults to "usability"
cf.run_metrics(PROJECT_PATH, subset="usability_annotation_set",
               metrics=[cf.exclusive_label_annotation(["usable", "not_usable"], name="usability",
                                                      requires_mask=False)])
usable_ids = cf.ids_matching(PROJECT_PATH, "usability", {"usability": "usable"})
cf.define_subset(PROJECT_PATH, name="usability_annotation_set_usable", occurrence_ids=usable_ids)

# create the part segmenter reference set
# an occurrence already in the reference set stays in it, whatever it is labelled now
reference_ids = cf.select_ids(PROJECT_PATH, subset="mask_reference_set")
cf.grow_subset(PROJECT_PATH, name="mask_reference_set", target_size=100,
               candidate_ids=sorted(set(usable_ids) | set(reference_ids)),
               candidate_note="labelled usable, plus everything already in the reference set")
cf.run_segments(
    PROJECT_PATH,
    from_part="organism",
    subset="mask_reference_set",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient()],
    outputs={part: [cf.correct_mask()] for part in ["head", "thorax", "abdomen"]},
    reference=True,
    force=False,
    visualize=False,
    batch_size=1,
)
