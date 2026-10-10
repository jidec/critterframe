import logging

import critterframe as cf
from critterframe.selection.subsets import select_ids
from critterframe.selection.queries import ids_with_mask
from critterframe.metrics.embedding import pretrained
from critterframe.extensions.smp_segmenter import segmentation

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# cf.run_segments(
#     PROJECT_PATH,
#     run_name="body_parts_tighter",
#     from_part="organism",
#     shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
#     outputs={
#         "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"),mask_threshold=1.0)],
#     },
# )
#
# cf.define_subset(PROJECT_PATH,name="completed_tight_abdomens",occurrence_ids=cf.ids_completed(PROJECT_PATH,run_name="body_parts_tighter"))
#
# cf.grow_subset(PROJECT_PATH, name="abdomen_annotations", target_size=200, from_subset="completed_tight_abdomens")
# cf.run_metrics(PROJECT_PATH, subset="abdomen_annotations",
#                part="abdomen",
#                transforms=[cf.remove_background(),
#                            cf.remove_islands(),
#                            cf.crop_to_mask(),
#                            cf.orient(axis_strategy="longer"),
#                            cf.erode(fraction=0.15)],
#                metrics=[cf.exclusive_label_annotation(labels=["perfect",
#                                                               "good",
#                                                               "poor"],
#                                                       note="perfect": the abdomen entire, not curved, and with <5% background bleed
#                                                            "good: representative of color pattern of abdomen but doesn't have to be entire abdomen and can be curved - for example half of abdomen containing all the colors that exist across the whole abdomen approximately proportional to whole abdomen colors\n"
#                                                            "poor: not representative of color pattern of abdomen, not of abdomen at all, has >5% background bleed \n",
#                                                       show_original=True,
#                                                             )])
#
#
# # left out for now: abdomen embeddings & clustering on them
# # cf.run_metrics(
# #     PROJECT_PATH, part="abdomen",
# #     run_name="abdomen_embeddings",
# #     subset="completed_tight_abdomens",
# #     transforms=[cf.remove_background(), cf.crop_to_mask(),
# #                 cf.orient(axis_strategy="longer")],
# #     metrics=[cf.embedding(pretrained("resnet18", resize="pad"),
# #                                 name="resnet18_embedding")],   # run_name defaults to "resnet18_embedding"
# # )
#
# # resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
# #                                 name="resnet18_embedding")
# # cf.run_metrics(
# #     PROJECT_PATH, run_name="abdomen_clusters", part="abdomen",
# #     subset="completed_tight_abdomens",
# #     metrics=[cf.cluster([resnet_embedding], from_run="abdomen_embeddings",
# #                         n_clusters=10, n_components=8)]
# # )
#
# cf.run_metrics(
#     PROJECT_PATH, run_name="abdomen_qc",
#     part="abdomen",
#     metrics=[cf.mask_info(operations=["segment"]),
#              cf.mask_fraction(), cf.blur_variance(),cf.mask_area(),cf.n_islands(),
#              cf.elongation(), cf.jaggedness()],
# )
#
cf.run_metrics(
    PROJECT_PATH,
    run_name="abdomen_color_thresholds",
    part="abdomen",
    transforms=[cf.remove_background(),
                   cf.remove_islands(),
                   cf.erode(fraction=0.15)],
    metrics=[
        cf.threshold_fractions([
            # LCh hue of reference sRGB colours: red 36, orange-red 46, orange 64,
            # amber 83, yellow 100, green 141, blue 297, violet 313, purple 321, magenta 339
            cf.color_threshold("red",        lch_h=(345, 50),  lch_c=(30, None)),
            cf.color_threshold("orange",     lch_h=(50, 75),   lch_c=(30, None)),
            cf.color_threshold("yellow",     lch_h=(75, 105),  lch_c=(30, None)),
            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
            cf.color_threshold("blue",       lch_h=(195, 305), lch_c=(30, None), lch_l=(None, 55)),
            cf.color_threshold("light_blue", lch_h=(195, 305), lch_c=(20, None), lch_l=(55, None)),
            cf.color_threshold("purple",     lch_h=(305, 345), lch_c=(30, None)),
        ], unmatched=True, name="color_bins"),
        # present at >= 5% of the abdomen; n_colors_present; ranked_color_1/2
        cf.color_presence(min_fraction=0.05, n_ranked_colors=2),
    ],
    force=True
)

filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="abdomen",
    metric_specs={
        "abdomen_qc": {
            "blur_variance": "below",  # out of focus
            "mask_fraction": "below",  # tiny within image
            #"mask_area": "below",  # tiny in general - wrong filter for the current annot set
            "n_islands": "above", # many fragments
            "elongation": "below", # not slender as an abdomen should be
            "jaggedness": "above", # abdomens shouldn't be too jagged
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="abdomen_annotations",
    max_fpr=0.1,
    min_precision=0.8,
    drop_redundant=True,     # a filter that catches nothing the others miss only costs good organisms
)

df = cf.export_metrics(
    PROJECT_PATH,
    run_names=["abdomen_color_thresholds"],
    parts=["abdomen"],
    filters=filters,
    rename={
        "abdomen_color_thresholds__abdomen__color_bins__red": "red_prop",
        "abdomen_color_thresholds__abdomen__color_bins__orange": "orange_prop",
        "abdomen_color_thresholds__abdomen__color_bins__yellow": "yellow_prop",
        "abdomen_color_thresholds__abdomen__color_bins__green": "green_prop",
        "abdomen_color_thresholds__abdomen__color_bins__blue": "blue_prop",
        "abdomen_color_thresholds__abdomen__color_bins__light_blue": "light_blue_prop",
        "abdomen_color_thresholds__abdomen__color_bins__purple": "purple_prop",
        "abdomen_color_thresholds__abdomen__color_bins__unmatched": "unmatched_prop",
        "abdomen_color_thresholds__abdomen__color_presence__red_present": "red_presence",
        "abdomen_color_thresholds__abdomen__color_presence__orange_present": "orange_presence",
        "abdomen_color_thresholds__abdomen__color_presence__yellow_present": "yellow_presence",
        "abdomen_color_thresholds__abdomen__color_presence__green_present": "green_presence",
        "abdomen_color_thresholds__abdomen__color_presence__blue_present": "blue_presence",
        "abdomen_color_thresholds__abdomen__color_presence__light_blue_present": "light_blue_presence",
        "abdomen_color_thresholds__abdomen__color_presence__purple_present": "purple_presence",
        "abdomen_color_thresholds__abdomen__color_presence__n_colors_present": "n_colors_present",
        "abdomen_color_thresholds__abdomen__color_presence__ranked_color_1": "ranked_color_1",
        "abdomen_color_thresholds__abdomen__color_presence__ranked_color_2": "ranked_color_2",
    },
    occurrence_columns=["sex","occurrence_id","eventDate","decimalLatitude","decimalLongitude",
                        "coordinateUncertaintyInMeters","order","family","genus","species","scientificName","image_url"],
)

# cf.render_segments(PROJECT_PATH,
#                    name="abdomen_renders_filtered",
#                    occurrence_ids=df["occurrence_id"].tolist()[:1000],
#                    part="abdomen",
#                    limit=1000,
#                    transforms=[cf.remove_background(),
#                            cf.remove_islands(),
#                            cf.crop_to_mask(),
#                            cf.orient(axis_strategy="longer"),
#                            cf.erode(fraction=0.15),
#                                cf.crop_to_mask(),
#                                cf.resize(20,100)])