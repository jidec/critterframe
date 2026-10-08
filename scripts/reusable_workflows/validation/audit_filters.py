"""
Do the export filters remove the bad part segments, and at what cost?

validate_masks.py says how accurate a part mask is where a reference mask could
be drawn, which is usable images only. This says what share of the rows an
export keeps are bad, over every input: thresholds are calibrated against the
screening labels in one sample, then the whole filter set is scored on a second
sample it never saw.

The numbers to report are the bad rate before and after filtering and the share
of rows kept, each with its interval. `kept_bad` and `dropped_good` are the ids
to look at when deciding what candidate score to add next.

Nothing is deleted: filters apply only at export.

Before: reference_annotation/annotate_segment_quality.py for the labels.
"""

import logging

import critterframe as cf
from critterframe.metrics.embedding import pretrained

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

PARTS = ["head", "thorax", "abdomen"]
UPSTREAM_PART = "organism"

QUALITY_RUN = "segment_quality"     # also the label's metric name; see annotate_segment_quality.py
CALIBRATE_SUBSET = "segment_quality_calibrate"
AUDIT_SUBSET = "segment_quality_audit"

PART_QC_RUN = "part_qc"
UPSTREAM_QC_RUN = "qc"
EMBEDDING_RUN = "part_embedding"
CLUSTER_RUN = "part_clusters"
SCORE_RUN = "part_bad_score"
MAX_FPR = 0.02

# 1. Candidate scores over the whole project. mask_info() is what the
#    segmentation run recorded on each mask (the segmenter's own score, an
#    unsure orientation); the upstream QC scores are candidates too, since a bad
#    organism mask takes every part below it down.
cf.run_metrics(PROJECT_PATH, run_name=PART_QC_RUN, parts=PARTS,
               metrics=[cf.mask_info(), cf.mask_fraction()])
cf.run_metrics(PROJECT_PATH, run_name=UPSTREAM_QC_RUN, part=UPSTREAM_PART,
               metrics=[cf.blur_variance(), cf.edge_fraction()])

#    Two candidates built on an embedding of each segment (needs the [torch]
#    extra). Clusters are what used to be picked by eye; here the worst are
#    dropped by the bad rate of their own labels. label_score skips the clusters
#    and fits the labels directly, on the calibration sample only.
part_embedding = cf.embedding(pretrained("resnet18", resize="pad"), name="embedding")
cf.run_metrics(PROJECT_PATH, run_name=EMBEDDING_RUN, parts=PARTS,
               transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient()],
               metrics=[part_embedding])
cf.run_metrics(PROJECT_PATH, run_name=CLUSTER_RUN, parts=PARTS,
               metrics=[cf.cluster([part_embedding], from_run=EMBEDDING_RUN,
                                   n_clusters=30, n_components=8)])
cf.run_metrics(PROJECT_PATH, run_name=SCORE_RUN, parts=PARTS,
               metrics=[cf.label_score([part_embedding], from_run=EMBEDDING_RUN,
                                       labels_run=QUALITY_RUN,
                                       labels_subset=CALIBRATE_SUBSET,
                                       label_metric=QUALITY_RUN,
                                       good_labels=["good"])])

for part in PARTS:
    # 2. Calibrate on one sample. Every candidate measured on this part goes in
    #    one call, keyed by the run it lives in. Each is held to MAX_FPR on its
    #    own; one that catches no bad label is left out and warned about, and
    #    what the chosen ones do together on these labels is logged.
    #    "category" drops the worst clusters within the same budget; a cluster
    #    with too few labels to judge is never dropped.
    filters = cf.get_validated_filters(
        PROJECT_PATH,
        {
            PART_QC_RUN: {"mask_info__segment__score": "below"},
            SCORE_RUN: {"label_score__bad_probability": "above"},
            CLUSTER_RUN: {"cluster__cluster_id": "category"},
        },
        annotation_run=QUALITY_RUN, label_metric=QUALITY_RUN,
        good_labels=["good"], part=part, subset=CALIBRATE_SUBSET,
        max_fpr=MAX_FPR,
    )
    #    The upstream scores were measured on another part, so they are a call
    #    of their own, judged against this part's labels.
    filters.update(cf.get_validated_filters(
        PROJECT_PATH, ["blur_variance", "edge_fraction"],
        predicted_run=UPSTREAM_QC_RUN, annotation_run=QUALITY_RUN,
        label_metric=QUALITY_RUN, good_labels=["good"],
        part=UPSTREAM_PART, label_part=part, subset=CALIBRATE_SUBSET,
        max_fpr=MAX_FPR,
    ))

    # 3. Audit the set as a whole on the other sample. Each filter was allowed
    #    MAX_FPR alone, so together they can cost more good rows than that;
    #    good_retained is where it shows. (With every candidate in one call,
    #    get_validated_filters(audit_subset=AUDIT_SUBSET) does this itself.)
    audit = cf.audit_filters(PROJECT_PATH, filters, annotation_run=QUALITY_RUN,
                             label_metric=QUALITY_RUN, good_labels=["good"],
                             part=part, subset=AUDIT_SUBSET)
    if not audit:
        continue

    logging.info("%s bad segments still kept: %s", part, audit["kept_bad"])
    logging.info("%s good segments removed: %s", part, audit["dropped_good"])

    # 4. Export this part with the filters that were audited.
    cf.export_metrics(PROJECT_PATH, parts=[part, UPSTREAM_PART], filters=filters)
