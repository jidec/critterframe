"""
Do the export filters remove bad abdomen segments?

The part reference masks (reference_annotations_and_masks.py) exist only for
images screened usable, so validate_masks says how accurate a part mask is on a
valid input and nothing about the invalid inputs that reach the part segmenters
anyway. Here a person screens a random sample of finished segments from the
whole population, and four candidate filters are calibrated against those
labels: the segmenter's own score, an outlier score over resnet embeddings,
clusters of those embeddings dropped by their labelled bad rate (what used to
be picked by eye), and a score fitted to the labels directly.

Calibration only for now, so every rate logged here describes the labels the
filters were chosen on. Stating how clean an export is needs cf.audit_filters
on a second sample, screened from outside part_screening_calibrate.

Second stage of a cascade: odonata_inat_obsorg_organism_gate.py has already
removed what is clearly invalid at the organism level, and everything here runs
on the occurrences that passed ("organism_gate_pass"). An export needs both
sets of filters, the gate's and the ones calibrated here.

Abdomens only for now.
"""

import logging

import critterframe as cf
from critterframe.metrics.embedding import pretrained
from critterframe.selection.subsets import select_ids
from critterframe.selection.queries import ids_with_mask
from critterframe.validation.filters import SEGMENT_BAD_LABELS

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# every gated occurrence with an abdomen mask, which is what an abdomen export holds
pool = (set(ids_with_mask(PROJECT_PATH, part="abdomen"))
        & set(select_ids(PROJECT_PATH, subset="organism_gate_pass")))

cf.grow_subset(PROJECT_PATH, "part_screening_calibrate", target_size=100,
               candidate_ids=sorted(pool))

# screen the abdomen in the organism crop its segmenter saw: the shared_steps
# odonata_inat_obsorg.py's body_parts run segments with
cf.run_metrics(
    PROJECT_PATH, run_name="segment_quality", part="abdomen",
    subset="part_screening_calibrate", from_part="organism",
    transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    metrics=[cf.segment_quality_annotation()], visualize=False,
)

# --- candidate scores, over everything that passed the gate --------------------

cf.run_metrics(PROJECT_PATH, run_name="part_qc", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.mask_info(), cf.mask_fraction()])

# The same recipe and name odonata_inat_obsorg.py embeds its 5000-abdomen subset
# with, so those are skipped and only the rest of the gated abdomens are embedded.
resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")
cf.run_metrics(
    PROJECT_PATH, part="abdomen", subset="organism_gate_pass",
    transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                cf.orient(axis_strategy="longer")],
    metrics=[resnet_embedding],   # run_name defaults to "resnet18_embedding"
)

cf.run_metrics(PROJECT_PATH, run_name="abdomen_outliers", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.outlier([resnet_embedding], from_run="resnet18_embedding")])

# 30 clusters: more than anyone would inspect by hand, and nobody has to
cf.run_metrics(PROJECT_PATH, run_name="abdomen_clusters", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.cluster([resnet_embedding], from_run="resnet18_embedding",
                                   n_clusters=10, n_components=16)])

# fitted on the screened sample's labels
cf.run_metrics(PROJECT_PATH, run_name="abdomen_bad_score", part="abdomen", subset="organism_gate_pass",
               metrics=[cf.label_score([resnet_embedding], from_run="resnet18_embedding",
                                       labels_run="segment_quality",
                                       labels_subset="part_screening_calibrate")])

# --- calibrate each candidate against the labels ------------------------------


def threshold(metric, flag_when, run):
    # max_fpr: the share of good abdomens one threshold may cost
    return cf.get_validated_filters(
        PROJECT_PATH, {metric: flag_when}, predicted_run=run,
        annotation_run="segment_quality", label_metric="segment_quality_annotation",
        bad_labels=SEGMENT_BAD_LABELS, part="abdomen",
        subset="part_screening_calibrate", max_fpr=0.05)


candidates = {
    "segmenter score": threshold("mask_info__segment__score", "below", "part_qc"),
    # IsolationForest's decision function: lower is more anomalous
    "outlier score": threshold("outlier__anomaly_score", "below", "abdomen_outliers"),
    # a cluster is dropped when more than 20% of its labels are bad
    "clusters": cf.get_validated_categories(
        PROJECT_PATH, "cluster__cluster_id", "abdomen_clusters", "segment_quality",
        part="abdomen", subset="part_screening_calibrate", max_bad_rate=0.2),
    "label score": threshold("label_score__bad_probability", "above", "abdomen_bad_score"),
}
candidates["all together"] = {column: rule for filters in candidates.values()
                              for column, rule in filters.items()}

# Each call above logs its own sweep (recall, false-positive rate per cutoff) or
# per-cluster table; these are the filters they arrived at.
for name, filters in candidates.items():
    logging.info("%s: %s", name, filters or "nothing calibrated")
