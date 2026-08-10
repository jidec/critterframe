"""
Light-trap insects preprocessed by Antenna

Antenna has already detected individual insects on each light-trap sheet and
cut a crop per detection, so two things differ from the generic pipeline:

  ingest comes from the Antenna extension, which maps Antenna's column names
  onto occurrence_id/image_url, drops the detections Antenna determined hold no
  moth, and recovers which sheet image each crop came from

  segmentation runs with detect_bounds=False. The image IS the organism's
  crop, so there's nothing left to detect -- SAM2 is prompted geometrically
  instead, and the Grounding DINO detector is never even loaded.

Long term this script is the one to run on an interval as new occurrences come
into Antenna. Every step is repeat-aware, so a scheduled rerun processes only
what's new.

The human review section is the exception, and is off by default: it opens
windows and blocks on a person, which a scheduled rerun has nobody to satisfy.
Turn it on when you sit down to review. What it produces -- labels and
corrected reference masks over a small frozen sample -- goes through the same
machinery as everything else, so it lands beside the automated results and is
compared with no special plumbing.

Needs ANTENNA_EMAIL, ANTENNA_PW, and ANTENNA_PROJECT_ID in the environment (a
.env file locally) for the API steps -- both the occurrence export and the
capture images the scale pass measures come through the same authenticated
session, so there are no separate cloud credentials to arrange.
"""

import logging
import critterframe as cf
from critterframe.extensions.antenna_lighttraps import ingest as antenna_ingest
from critterframe.project.subsets import select_ids
from critterframe.extensions.antenna_lighttraps.calibrations import (  # noqa: E402
    scale as antenna_scale,
)

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/GitProjects/cf_projects/mm2_antenna_insects"
EXPORT_CSV = "C:/new_new_downloads/moth-monitoring-20_export-121 (1).csv"

# ingest through Antenna extension
antenna_ingest.ingest_occurrences(PROJECT_PATH, import_csv_path=EXPORT_CSV)

# download images using urls in ingest
cf.download_images(PROJECT_PATH)

# detect_bounds=False because the crop is already the organism.
# use a center point, which works well for Antenna crops
# branching of retry without center isn't quite consistent with CritterFrames, revise later
cf.run_segments(
    PROJECT_PATH,
    steps=[
        cf.segment(cf.groundedsam2(detect_bounds=False,use_center_point=True)),
    ],
)

# run traits after removing appendages and orienting
cf.run_metrics(
    PROJECT_PATH,
    run_name="traits",
    transforms=[
        cf.remove_appendages(),
        cf.orient(),
    ],
    metrics=[
        cf.body_length(),
        cf.max_width(),
        cf.mask_area(name="area_px", unit="px2"),
    ],
)

# BOOM, above is all you need for traits
# the below more advanced code is for validation and filtering (including validation set creation) specific to Antenna data
# as well as the final export

# run metrics used for quality control filtering
cf.run_metrics(
    PROJECT_PATH,
    run_name="qc",
    metrics=[
        cf.edge_fraction(),
    ],
)
# bilateral_asymmetry needs both transforms
cf.run_metrics(
    PROJECT_PATH,
    run_name="qc",
    transforms=[
        cf.remove_appendages(),
        cf.orient(),
    ],
    metrics=[
        cf.bilateral_asymmetry(),
    ],
)

# Human review, in two passes: screen the sample first, then make reference data
# only for the crops a reference is definable for. The order is the whole point.
# Asking someone to correct the mask on a crop holding two moths, no moth, or
# half a moth running off the frame has no right answer -- there is no single
# complete organism whose boundary the segmenter could have got right or wrong --
# so they invent one, it lands in reference_masks.parquet, and it drags down the
# IoU validate_masks reports. The screening flag is what says which crops the
# expensive human questions can be asked of at all.

# pass 1: screen the sample. EVERY crop in it, including the ones that turn out
# unusable -- get_validated_filters below scores edge_fraction and
# bilateral_asymmetry at catching flagged crops, and a sample with the flagged
# ones removed has nothing to detect and can't be scored at all.

REVIEW_TARGET = 50  # total screening sample size; raise to grow
try:
    already_reviewed = select_ids(PROJECT_PATH, subset="review")
except KeyError:  # first run, subsets.toml has no "review" yet
    already_reviewed = []

candidates = [i for i in select_ids(PROJECT_PATH) if i not in set(already_reviewed)]
cf.define_subset(
    PROJECT_PATH,
    "review",
    occurrence_ids=sorted(set(already_reviewed) | set(
        cf.sample_occurrences(candidates, max(0, REVIEW_TARGET - len(already_reviewed))))),
)
cf.run_metrics(
    PROJECT_PATH,
    run_name="human_annotation_labels",
    subset="review",
    metrics=[
        cf.annotate_flags(),
    ],
)

# The crops a reference is even definable for. Defined by the HUMAN flag, never
# by whether the automated filter passed them: those are independent on purpose.
# The segmenter is graded on the crops a person called usable, and the filter is
# graded against that same person's flags -- draw the mask sample from "whatever
# the filter let through" instead and every filter mistake becomes invisible to
# both. On a fresh project nobody has screened yet this selects nothing and the
# two passes below process nothing, which is correct rather than broken.
cf.define_subset(
    PROJECT_PATH,
    "reference",
    occurrence_ids=cf.occurrences_matching(
        PROJECT_PATH, "human_annotation_labels",
        {"annotate_flags": "usable"}),
)

# pass 2: reference data, over those only -- correct the mask and click the axis
# in one visit, since both are being asked of the same crops.
# from_part puts SAM2's mask in front of the person to correct.
# reference=True writes to reference_masks.parquet, so these coexist with the
# canonical masks rather than replacing them
cf.run_segments(
    PROJECT_PATH,
    run_name="human_corrected",
    subset="reference",
    from_part=cf.DEFAULT_PART,
    steps=[cf.correct_mask()],
    reference=True,
)
cf.run_metrics(
    PROJECT_PATH,
    run_name="human_measurements",
    subset="reference",
    metrics=[
        cf.click_two_points(labels=("head","tail")),
    ],
)

# validate the reference masks against the automated masks
# make sure transforms line up with how you're doing the reference
# in this case, the reference mask is really the sam2 mask with appendages removed, so that's what we're comparing here
# The population is wherever a reference mask exists, so what this reports is
# IoU over the crops a human called usable -- which is the honest number, and
# the one that describes the exported dataset, since the export filters to
# approximately that same population.
cf.validate_masks(PROJECT_PATH, transforms=[cf.remove_appendages()])

# compare automated body length to length obtained by clicking points manually (the reference)
cf.compare_metrics(
    PROJECT_PATH, "traits", "human_measurements",
    metric_names={"body_length": "click_two_points__length_px"},
)

# measure physical scale once per event (one trap night)
antenna_scale.measure_scales(PROJECT_PATH)

# calibrate the QC cutoffs against the human flags: every observed value of each
# metric is scored as a candidate cutoff, and the one kept is the highest-recall
# cutoff that still throws away no more than X% (2% default) of the crops a person called
# usable. The sweeps are logged in full, so the evidence behind each choice is
# in the run output. This reads the screening run, which covers the whole sample
# -- both classes present, which is what makes a cutoff scorable.
filters = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs=["edge_fraction", "bilateral_asymmetry"],
    predicted_run="qc",
    annotation_run="human_annotation_labels",
    max_fpr=0.20,   # "Don't throw away more than 10% of good data"
    min_precision=0.5, # "at least 50% of what I exclude should genuinely be bad"
)

# Export, excluding the occurrences those filters flag. Nothing is deleted --
# every excluded occurrence keeps its mask and its measurements, so a threshold
# can be revised and this rerun. A filter naming a column no run produced raises
# rather than being ignored, so adding blur_variance here means adding
# cf.blur_variance() to the qc run first.
cf.export_metrics(
    PROJECT_PATH,
    f"{PROJECT_PATH}/traits.csv",
    occurrence_columns=["determination_name", "first_appearance_timestamp",
                        "event_name", "deployment_name"],
    filters=filters,
    units="mm",   # px traits divided by that night's scale, px_per_mm alongside
)

cf.print_summary(PROJECT_PATH)