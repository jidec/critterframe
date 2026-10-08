"""
A conservative gate after the organism segment: which occurrences are worth
running parts on.

The organism segment says things about validity the part crops can't: how sure
the detector was, whether it saw a second organism, whether the mask runs off
the frame. Cutoffs on those scores are calibrated against the usability labels
and the occurrences that pass are frozen as the subset "organism_gate_pass",
which the part segmentation and the abdomen filter validation then run on.

Conservative on purpose: each cutoff may cost at most 2% of usable images, so
this removes what is clearly invalid and leaves the rest to the filters at the
end. A good specimen gated out here never gets parts, and nothing downstream
would notice.

Nothing is deleted. The gate only decides where later steps spend effort;
loosening it and rerunning adds the newly admitted occurrences. The same
filters still belong in the export's `filters`, which is what records them.

Every rate logged here is measured on the labels the cutoffs were chosen on.
"""

import logging

import critterframe as cf
from critterframe.validation.filters import BAD_LABELS

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# more usability labels: a 2% budget needs more than the 28 bad ones the first
# 100 held. Additive, so everything already screened is kept and skipped.
cf.grow_subset(PROJECT_PATH, name="usability_annotation_set", target_size=300)

cf.run_metrics(PROJECT_PATH, subset="usability_annotation_set",
               metrics=[cf.usability_annotation()])

# organism scores over the whole project. mask_info is what groundedsam2
# recorded on each mask: box_score, score, n_boxes, second_box_score, ...
cf.run_metrics(
    PROJECT_PATH, run_name="organism_qc",
    metrics=[cf.mask_info(operations=["segment"]), cf.edge_fraction(),
             cf.mask_fraction(), cf.blur_variance()],
)

# one cutoff per score, each allowed to cost 2% of usable images
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
    max_fpr=0.02,
)


def on_the_labels(filters):
    return cf.audit_filters(PROJECT_PATH, filters, annotation_run="usability_annotation",
                            label_metric="usability_annotation", bad_labels=BAD_LABELS,
                            subset="usability_annotation_set",
                            run_names=["organism_qc", "usability_annotation"])


# A score that can't separate the labels still comes back with a cutoff, at the
# edge of what the labelled images happened to span, and that cutoff would
# remove unlabelled occurrences beyond it on no evidence. Keep only the cutoffs
# that removed at least one unusable image.
caught = on_the_labels(gate)["filters"].set_index("filter")["removed_bad"]
gate = {column: rule for column, rule in gate.items() if caught[column] > 0}

for column, rule in gate.items():
    logging.info("gate: %s %s", column, rule)

# what the cutoffs do together, and which reasons they can't see: expect
# bad_angle, wrong_life_stage and dead to pass through to the end filters
together = on_the_labels(gate)
logging.info("gate keeps %.1f%% of usable images and removes %.1f%% of unusable ones",
             100 * together["good_retained"], 100 * together["bad_caught"])
logging.info("removed per reason: %s",
             {flag: f"{together[f'recall_{flag}']:.0%} of {together[f'n_{flag}']}"
              for flag in BAD_LABELS if together[f"n_{flag}"]})
logging.info("per cutoff:\n%s", together["filters"].to_string(index=False))

# freeze the occurrences that pass, redefined on every run so a recalibrated
# gate takes effect
measured = cf.export_metrics(PROJECT_PATH, path=False, manifest=False,
                             run_names=["organism_qc"], parts=["organism"])
passing = cf.export_metrics(PROJECT_PATH, path=False, manifest=False,
                            run_names=["organism_qc"], parts=["organism"],
                            filters=gate)

cf.define_subset(PROJECT_PATH, name="organism_gate_pass",
                 occurrence_ids=sorted(passing["occurrence_id"]),
                 note=f"passes the organism gate: {gate}")

logging.info("organism gate: %d of %d measured occurrences pass (%.1f%%)",
             len(passing), len(measured), 100 * len(passing) / max(1, len(measured)))
