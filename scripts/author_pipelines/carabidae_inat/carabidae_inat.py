"""
Carabid beetles from iNaturalist, ingested from a GBIF archive.

Ingest with cross-source deduplication and a cap of 500 occurrences per
species, download the images, and segment the organism.
"""

import logging

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/carabidae_inat"

gbif_ingest.ingest_occurrences(
    PROJECT_PATH,
    archive_path="D:/0013055-260928105237408.zip",
    occurrence_columns=gbif_ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS,
    inat_photo_size="medium",
    dedupe_key_cols=["eventDate","decimalLatitude","decimalLongitude"],
    group_col="scientificName",
    max_per_group=500,
    trust_source_file_unchanged=True,
    prioritize_inat=True,
)

cf.download_images(PROJECT_PATH)

cf.run_segments(
    PROJECT_PATH,
    run_name="organism_groundedsam2",
    steps=[
        cf.segment(cf.groundedsam2(text_prompt="insect.", box_threshold=0.25, text_threshold=0.25),
                   mask_threshold=0.0),
    ],
    visualize_every=50,
)

# each part is shown as the colour traits see it: islands removed and the edge eroded
# cf.grow_subset(PROJECT_PATH, name="quality_annotations_test", target_size=100)
# cf.run_metrics(PROJECT_PATH, subset="quality_annotations_test",
#            transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
#                        cf.orient(), cf.crop_to_mask(),cf.erode(),cf.crop_inside_mask()],
#            metrics=[cf.exclusive_label_annotation(
#                         labels=["good", "poor"],
#                         note="These are judgements about segment shape, NOT illumination.\n"
#                              "good: the part entire, not curved, not covered by wing, and with <5% background bleed\n",
#                         show_original=True)],
#                force=True)

cf.run_metrics(PROJECT_PATH, run_name="color_thresholds",
               transforms=[cf.remove_background(), cf.remove_islands()],
               metrics=[cf.threshold_fractions([
                            cf.color_threshold("red",        lch_h=(345, 50),  lch_c=(30, None)),
                            cf.color_threshold("orange",     lch_h=(50, 75),   lch_c=(30, None)),
                            cf.color_threshold("yellow",     lch_h=(75, 105),  lch_c=(30, None)),
                            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
                            cf.color_threshold("blue",       lch_h=(195, 305), lch_c=(30, None), lch_l=(None, 55)),
                            cf.color_threshold("light_blue", lch_h=(195, 305), lch_c=(20, None), lch_l=(55, None)),
                            cf.color_threshold("purple",     lch_h=(305, 345), lch_c=(30, None)),
                        ], unmatched=True, name="color_bins"),
                        # present at >= 5% of the part; n_colors_present; ranked_color_1/2
                        cf.color_presence(min_fraction=0.05, n_ranked_colors=2),
                    ])

# export
cf.export_metrics(
    PROJECT_PATH,
    run_names=["color_thresholds"],
    rename={
        "color_thresholds__organism__color_bins__red": "red_prop",
        "color_thresholds__organism__color_bins__orange": "orange_prop",
        "color_thresholds__organism__color_bins__yellow": "yellow_prop",
        "color_thresholds__organism__color_bins__green": "green_prop",
        "color_thresholds__organism__color_bins__blue": "blue_prop",
        "color_thresholds__organism__color_bins__light_blue": "light_blue_prop",
        "color_thresholds__organism__color_bins__purple": "purple_prop",
        "color_thresholds__organism__color_bins__unmatched": "unmatched_prop",
        "color_thresholds__organism__color_presence__red_present": "red_presence",
        "color_thresholds__organism__color_presence__orange_present": "orange_presence",
        "color_thresholds__organism__color_presence__yellow_present": "yellow_presence",
        "color_thresholds__organism__color_presence__green_present": "green_presence",
        "color_thresholds__organism__color_presence__blue_present": "blue_presence",
        "color_thresholds__organism__color_presence__light_blue_present": "light_blue_presence",
        "color_thresholds__organism__color_presence__purple_present": "purple_presence",
        "color_thresholds__organism__color_presence__n_colors_present": "n_colors_present",
        "color_thresholds__organism__color_presence__ranked_color_1": "ranked_color_1",
        "color_thresholds__organism__color_presence__ranked_color_2": "ranked_color_2",
    },
    occurrence_columns=["sex", "occurrence_id", "eventDate", "decimalLatitude", "decimalLongitude",
                        "coordinateUncertaintyInMeters", "order", "family", "genus", "species",
                        "scientificName", "image_url"],
)