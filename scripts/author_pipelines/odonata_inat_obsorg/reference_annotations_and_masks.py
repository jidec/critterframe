import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# grow a subset of occurrence images intended to get usability annotations
cf.grow_subset(PROJECT_PATH, name="usability_annotation_set", target_size=55)

cf.run_metrics(PROJECT_PATH, subset="usability_annotation_set",
               metrics=[cf.usability_annotation()])

# define a subset of usable occurrence images
cf.define_subset(
    PROJECT_PATH, name="usability_annotation_set_usable",
    occurrence_ids=cf.occurrences_matching(
        PROJECT_PATH, "usability_annotation", {"usability_annotation": "usable"}),
)

# grow a subset of occurrence images intended to get reference masks
cf.grow_subset(PROJECT_PATH, name="mask_reference_set", target_size=50,
               from_subset="usability_annotation_set_usable")

# run head, thorax, abdomen reference sets
cf.run_segments(
    PROJECT_PATH,
    from_part="organism",
    subset="mask_reference_set_with_previous",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient()],
    outputs={part: [cf.correct_mask()] for part in ["head","thorax","abdomen"]},
    reference=True,
    force=False,
    visualize=False,
    batch_size=1,
)