"""
Step 7 of 8: calibrate each part's filters against the screening labels and
export one table.

Each part is filtered on its own: an occurrence whose abdomen fails keeps its
head and thorax values.

Unattended.

Before: 6_annotate_parts_for_filters.py. Next: 8_render_filtered_parts.py.
"""

import logging

import critterframe as cf
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# a starting point copied from the thorax; edit once head labels exist
head_filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="head",
    metric_specs={
        "qc": {
            "mask_info__segment__score": "below",  # mask not confident
            "blur_variance": "below",              # out of focus
            "mask_fraction": "below",              # tiny within image
            "mask_area": "below",                  # tiny in general
            "n_islands": "above",                  # many fragments
            "elongation": "above",                 # too slender for a roughly round head
            "jaggedness": "above",                 # shouldn't be too jagged
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="head_annotations",
    max_fpr=0.1,
    min_precision=0.8,
    drop_redundant=True,     # a filter that catches nothing the others miss only costs good segments
)

thorax_filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="thorax",
    metric_specs={
        "qc": {
            "mask_info__segment__score": "below",  # mask not confident
            "blur_variance": "below",              # out of focus
            "mask_fraction": "below",              # tiny within image
            "mask_area": "below",                  # tiny in general
            "n_islands": "above",                  # many fragments
            "elongation": "above",                 # too slender for a roughly round thorax
            "jaggedness": "above",                 # shouldn't be too jagged
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="thorax_annotations",
    max_fpr=0.1,
    min_precision=0.8,
    drop_redundant=True,
)

abdomen_filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="abdomen",
    metric_specs={
        "qc": {
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
    run_names=["color_thresholds"],
    part_filters={
        "head": head_filters,
        "thorax": thorax_filters,
        "abdomen": abdomen_filters,
    },
    rename={
        "color_thresholds__head__color_bins__red": "head_red_prop",
        "color_thresholds__head__color_bins__orange": "head_orange_prop",
        "color_thresholds__head__color_bins__yellow": "head_yellow_prop",
        "color_thresholds__head__color_bins__green": "head_green_prop",
        "color_thresholds__head__color_bins__blue": "head_blue_prop",
        "color_thresholds__head__color_bins__light_blue": "head_light_blue_prop",
        "color_thresholds__head__color_bins__purple": "head_purple_prop",
        "color_thresholds__head__color_bins__unmatched": "head_unmatched_prop",
        "color_thresholds__head__color_presence__red_present": "head_red_presence",
        "color_thresholds__head__color_presence__orange_present": "head_orange_presence",
        "color_thresholds__head__color_presence__yellow_present": "head_yellow_presence",
        "color_thresholds__head__color_presence__green_present": "head_green_presence",
        "color_thresholds__head__color_presence__blue_present": "head_blue_presence",
        "color_thresholds__head__color_presence__light_blue_present": "head_light_blue_presence",
        "color_thresholds__head__color_presence__purple_present": "head_purple_presence",
        "color_thresholds__head__color_presence__n_colors_present": "head_n_colors_present",
        "color_thresholds__head__color_presence__ranked_color_1": "head_ranked_color_1",
        "color_thresholds__head__color_presence__ranked_color_2": "head_ranked_color_2",

        "color_thresholds__thorax__color_bins__red": "thorax_red_prop",
        "color_thresholds__thorax__color_bins__orange": "thorax_orange_prop",
        "color_thresholds__thorax__color_bins__yellow": "thorax_yellow_prop",
        "color_thresholds__thorax__color_bins__green": "thorax_green_prop",
        "color_thresholds__thorax__color_bins__blue": "thorax_blue_prop",
        "color_thresholds__thorax__color_bins__light_blue": "thorax_light_blue_prop",
        "color_thresholds__thorax__color_bins__purple": "thorax_purple_prop",
        "color_thresholds__thorax__color_bins__unmatched": "thorax_unmatched_prop",
        "color_thresholds__thorax__color_presence__red_present": "thorax_red_presence",
        "color_thresholds__thorax__color_presence__orange_present": "thorax_orange_presence",
        "color_thresholds__thorax__color_presence__yellow_present": "thorax_yellow_presence",
        "color_thresholds__thorax__color_presence__green_present": "thorax_green_presence",
        "color_thresholds__thorax__color_presence__blue_present": "thorax_blue_presence",
        "color_thresholds__thorax__color_presence__light_blue_present": "thorax_light_blue_presence",
        "color_thresholds__thorax__color_presence__purple_present": "thorax_purple_presence",
        "color_thresholds__thorax__color_presence__n_colors_present": "thorax_n_colors_present",
        "color_thresholds__thorax__color_presence__ranked_color_1": "thorax_ranked_color_1",
        "color_thresholds__thorax__color_presence__ranked_color_2": "thorax_ranked_color_2",

        "color_thresholds__abdomen__color_bins__red": "abdomen_red_prop",
        "color_thresholds__abdomen__color_bins__orange": "abdomen_orange_prop",
        "color_thresholds__abdomen__color_bins__yellow": "abdomen_yellow_prop",
        "color_thresholds__abdomen__color_bins__green": "abdomen_green_prop",
        "color_thresholds__abdomen__color_bins__blue": "abdomen_blue_prop",
        "color_thresholds__abdomen__color_bins__light_blue": "abdomen_light_blue_prop",
        "color_thresholds__abdomen__color_bins__purple": "abdomen_purple_prop",
        "color_thresholds__abdomen__color_bins__unmatched": "abdomen_unmatched_prop",
        "color_thresholds__abdomen__color_presence__red_present": "abdomen_red_presence",
        "color_thresholds__abdomen__color_presence__orange_present": "abdomen_orange_presence",
        "color_thresholds__abdomen__color_presence__yellow_present": "abdomen_yellow_presence",
        "color_thresholds__abdomen__color_presence__green_present": "abdomen_green_presence",
        "color_thresholds__abdomen__color_presence__blue_present": "abdomen_blue_presence",
        "color_thresholds__abdomen__color_presence__light_blue_present": "abdomen_light_blue_presence",
        "color_thresholds__abdomen__color_presence__purple_present": "abdomen_purple_presence",
        "color_thresholds__abdomen__color_presence__n_colors_present": "abdomen_n_colors_present",
        "color_thresholds__abdomen__color_presence__ranked_color_1": "abdomen_ranked_color_1",
        "color_thresholds__abdomen__color_presence__ranked_color_2": "abdomen_ranked_color_2",
    },
    occurrence_columns=["sex", "occurrence_id", "eventDate", "decimalLatitude", "decimalLongitude",
                        "coordinateUncertaintyInMeters", "order", "family", "genus", "species",
                        "scientificName", "image_url"],
)
