"""
Step 4 of 6: a conservative gate on the organism segment, then head, thorax and
abdomen on what passes it.

Unattended.

Before: 2_annotate_references.py for the usability labels, and
3_train_part_segmenters.py for the models. Next: 5_screen_abdomens.py.
"""

import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# --- organism gate: which occurrences are worth running parts on --------------
#
# The organism segment says things about validity the part crops can't: how sure
# the detector was, whether it saw a second organism, whether the mask runs off
# the frame. Conservative on purpose: a good specimen gated out never gets
# parts, and nothing downstream would notice. Nothing is deleted, and loosening
# the gate later only adds the newly admitted occurrences.

# mask_info is what groundedsam2 recorded on each mask: box_score, score,
# n_boxes, second_box_score, ...
cf.run_metrics(
    PROJECT_PATH, run_name="organism_qc",
    metrics=[cf.mask_info(operations=["segment"]), cf.edge_fraction(),
             cf.mask_fraction(), cf.blur_variance(), cf.mask_area()],
)

# One cutoff per score, each allowed to cost 2% of usable images. A score that
# catches no unusable image is left out, and what the cutoffs do together is
# logged, along with the reasons they can't see: expect bad_angle,
# wrong_life_stage and dead to pass through to the abdomen filters.
gate = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs={
        "mask_info__segment__box_score": "below",          # detector unsure
        "mask_info__segment__score": "below",              # SAM2 unsure of its mask
        "mask_info__segment__second_box_score": "above",   # a second organism in frame
        "edge_fraction": "above",                          # mask runs off the frame
        "blur_variance": "below",                          # out of focus
    },
    predicted_run="organism_qc",
    annotation_run="usability_annotation",
    subset="usability_annotation_set",
    max_fpr=0.02,
)

passing = cf.export_metrics(PROJECT_PATH, path=False, manifest=False,
                            run_names=["organism_qc"], parts=["organism"], filters=gate)

cf.define_subset(PROJECT_PATH, name="organism_gate_pass",
                 occurrence_ids=sorted(passing["occurrence_id"]),
                 note=f"passes the organism gate: {gate}")

# # --- parts, on what passed the gate -------------------------------------------
#
cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    subset="organism_gate_pass",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "head":    [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"))],
        "thorax":  [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"))],
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
    },
)
#
# cf.run_segments(PROJECT_PATH, part="body", from_part=["head", "thorax", "abdomen"], steps=[],
#                 subset="organism_gate_pass")
