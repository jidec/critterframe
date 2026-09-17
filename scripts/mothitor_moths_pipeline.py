import logging
import cv2
import critterframe as cf
from critterframe.extensions.antenna_lighttraps import ingest as antenna_ingest
from critterframe.extensions.antenna_lighttraps.calibrations import (  # noqa: E402
    scale as antenna_scale,
)

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/mothitor_antenna_insects"

# ingest through Antenna extension
antenna_ingest.ingest_occurrences(PROJECT_PATH, import_csv_path="C:/new_new_downloads\insect-ecology-lab-mothitor-test_export-147.csv")

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

# add scale
cf.measure_scale_by_hand(PROJECT_PATH, cv2.imread(PROJECT_PATH + "/ama_2025-10-06_23_50_02_extra.jpg"), target_mm=10.0)  # applies to every occurrence

cf.print_summary(PROJECT_PATH)
cf.export_metrics(PROJECT_PATH,path="D:/GitProjects/cf_projects/mothitor_antenna_insects/metrics.csv",
                  occurrence_columns=["event_id","deployment_id","deployment_name","best_machine_prediction_name"])
# BOOM, above is all you need for traits
# the below more advanced code is for validation and filtering (including validation set creation) specific to Antenna data
# as well as the final export

# run metrics used for quality control filtering
cf.run_metrics(
    PROJECT_PATH,
    run_name="qc_filters",
    metrics=[
        cf.edge_fraction(),
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
cf.grow_subset(PROJECT_PATH, name="reference_set", target_size=50,
               from_subset="usability_annotation_set_usable")

cf.run_segments(
    PROJECT_PATH,
    subset="reference_set",
    from_part=cf.DEFAULT_PART,
    steps=[cf.correct_mask()],
    reference=True,
    force=False,
)

cf.run_metrics(
    PROJECT_PATH,
    run_name="manual_click_head_tail",
    subset="reference_set",
    metrics=[
        cf.click_two_points(labels=("head","tail")),
    ],
)

cf.validate_masks(PROJECT_PATH, transforms=[cf.remove_appendages()])

# compare automated body length to length obtained by clicking points manually (the reference)
cf.compare_metrics(
    PROJECT_PATH, "traits", "manual_click_head_tail",
    metric_names={"body_length": "click_two_points__length_px"},
)

# calibrate qc cutoffs
filters = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs=["edge_fraction"],
    predicted_run="qc_filters",
    annotation_run="usability_annotation_set",
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

