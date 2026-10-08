import logging

import critterframe as cf
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

abdomen_filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="abdomen",
    metric_specs={
        "abdomen_qc": {
            "blur_variance": "below",              # out of focus
            "mask_fraction": "below",              # tiny within image
            # "mask_area": "below",                # tiny in general - wrong filter for the current annot set
            "n_islands": "above",                  # many fragments
            "elongation": "below",                 # not slender as an abdomen should be
            "jaggedness": "above",                 # abdomens shouldn't be too jagged
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="abdomen_annotations",
    max_fpr=0.1,
    min_precision=0.8,
    drop_redundant=True,
)

# export
cf.export_metrics(
    PROJECT_PATH,
    path=exports_dir(PROJECT_PATH) / "part_colors.csv",   # 8_render_filtered_parts.py reads this
    run_names=["abdomen_color_thresholds"],
    filters= abdomen_filters,
    rename={
        "abdomen_color_thresholds__abdomen__color_bins__red": "abdomen_red_prop",
        "abdomen_color_thresholds__abdomen__color_bins__orange": "abdomen_orange_prop",
        "abdomen_color_thresholds__abdomen__color_bins__yellow": "abdomen_yellow_prop",
        "abdomen_color_thresholds__abdomen__color_bins__green": "abdomen_green_prop",
        "abdomen_color_thresholds__abdomen__color_bins__blue": "abdomen_blue_prop",
        "abdomen_color_thresholds__abdomen__color_bins__light_blue": "abdomen_light_blue_prop",
        "abdomen_color_thresholds__abdomen__color_bins__purple": "abdomen_purple_prop",
        "abdomen_color_thresholds__abdomen__color_bins__unmatched": "abdomen_unmatched_prop",
        "abdomen_color_thresholds__abdomen__color_presence__red_present": "abdomen_red_presence",
        "abdomen_color_thresholds__abdomen__color_presence__orange_present": "abdomen_orange_presence",
        "abdomen_color_thresholds__abdomen__color_presence__yellow_present": "abdomen_yellow_presence",
        "abdomen_color_thresholds__abdomen__color_presence__green_present": "abdomen_green_presence",
        "abdomen_color_thresholds__abdomen__color_presence__blue_present": "abdomen_blue_presence",
        "abdomen_color_thresholds__abdomen__color_presence__light_blue_present": "abdomen_light_blue_presence",
        "abdomen_color_thresholds__abdomen__color_presence__purple_present": "abdomen_purple_presence",
        "abdomen_color_thresholds__abdomen__color_presence__n_colors_present": "abdomen_n_colors_present",
        "abdomen_color_thresholds__abdomen__color_presence__ranked_color_1": "abdomen_ranked_color_1",
        "abdomen_color_thresholds__abdomen__color_presence__ranked_color_2": "abdomen_ranked_color_2",
    },
    occurrence_columns=["sex", "occurrence_id", "eventDate", "decimalLatitude", "decimalLongitude",
                        "coordinateUncertaintyInMeters", "order", "family", "genus", "species",
                        "scientificName", "image_url"],
)
