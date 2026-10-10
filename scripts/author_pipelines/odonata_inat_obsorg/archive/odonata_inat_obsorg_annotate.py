"""
Everything a person does at a window for odonata_inat_obsorg.py, in the order
it becomes possible. Each pass resumes: rerun and only what is unlabelled is
shown. See odonata_inat_obsorg.py for where this sits in the run order.

  1. usability labels        needs images only
  2. reference part masks    needs the organism segment
  3. abdomen screening       needs the organism gate and the part segments

The third is skipped until the pipeline has produced what it needs.
"""

import logging

import critterframe as cf
from critterframe.selection.subsets import select_ids
from critterframe.selection.queries import ids_with_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

cf.grow_subset(PROJECT_PATH, name="usability_annotation_set", target_size=150)

cf.run_metrics(PROJECT_PATH, subset="usability_annotation_set",
               metrics=[cf.usability_annotation()])

cf.define_subset(
    PROJECT_PATH, name="usability_annotation_set_usable",
    occurrence_ids=cf.ids_matching(
        PROJECT_PATH, "usability_annotation", {"usability_annotation": "usable"}),
)

cf.grow_subset(PROJECT_PATH, name="mask_reference_set", target_size=100,
               from_subset="usability_annotation_set_usable")

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

# --- 3. abdomen screening ------------------------------------------------------
#
# A random sample of finished abdomen segments, each given one key: good, or
# why it should not reach an export. Unlike a reference mask this label exists
# for every input, so it is what the abdomen filters are calibrated against.

abdomens = set(ids_with_mask(PROJECT_PATH, part="abdomen"))
gated = set(select_ids(PROJECT_PATH, subset="organism_gate_pass")) \
    if "organism_gate_pass" in cf.load_subsets(PROJECT_PATH) else set()

if not abdomens & gated:
    logger.info("no gated abdomen segments yet -- run odonata_inat_obsorg.py through "
                "the part segmentation, then come back to screen them")
else:
    cf.grow_subset(PROJECT_PATH, "part_screening_calibrate", target_size=100,
                   candidate_ids=sorted(abdomens & gated))

    # in the organism crop the abdomen segmenter saw: the shared_steps
    # odonata_inat_obsorg.py's body_parts run segments with
    cf.run_metrics(
        PROJECT_PATH, run_name="segment_quality", part="abdomen",
        subset="part_screening_calibrate", from_part="organism",
        transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
        metrics=[cf.segment_quality_annotation()], visualize=False,
    )
