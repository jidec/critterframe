"""
Which automated QC cutoffs actually separate good images from bad ones?
Calibrate them against the human usability labels, then export with them.

Each QC metric's observed values are swept as candidate cutoffs and scored
against the labels. The chosen cutoff is the one catching the most bad images
while wrongly discarding at most MAX_FPR of the good ones. A metric that can't
meet that is left out and warned about rather than given an invented
threshold. One figure per metric (recall, false-positive rate and precision
across cutoffs) is written under visualizations/pipeline/.

Nothing is deleted: filters apply only at export, so revising a cutoff later
costs one re-export.

Before: reference_annotation/annotate_usability.py for the labels.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
PART = "organism"

QC_RUN = "qc"
ANNOTATION_RUN = "usability_annotation"
MAX_FPR = 0.02

# 1. QC metrics over the whole project -- cheap, and the export filters every
#    occurrence by them, not just the labelled ones.
cf.run_metrics(
    PROJECT_PATH, run_name=QC_RUN, part=PART,
    metrics=[cf.blur_variance(), cf.edge_fraction(), cf.mask_fraction(),
             cf.bilateral_asymmetry()],
)

# 2. Calibrate. Each spec says which side means "flag this": blurrier images
#    have LOWER variance, cut-off and lopsided masks score HIGHER.
filters = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs={"blur_variance": "below", "edge_fraction": "above",
                  "bilateral_asymmetry": "above"},
    predicted_run=QC_RUN, annotation_run=ANNOTATION_RUN, part=PART,
    max_fpr=MAX_FPR,
)

# mask_fraction fails in BOTH directions (a speck of dirt, or the substrate
# instead of the organism), which one swept cutoff can't express. Bound it by
# hand; read the values off the qc histogram rather than trusting these.
# A named function rather than a lambda: the export manifest records a
# predicate by name, and every lambda is called "<lambda>".
def mask_fraction_between_0_005_and_0_8(values):
    return values.between(0.005, 0.8)


filters[f"{QC_RUN}__{PART}__mask_fraction"] = mask_fraction_between_0_005_and_0_8

for column, rule in filters.items():
    logging.info("filter: %s %s", column, rule)

# 3. Export with them. An empty dict means nothing could be calibrated and the
#    export is unfiltered -- the log above says why.
cf.export_metrics(PROJECT_PATH, filters=filters)
