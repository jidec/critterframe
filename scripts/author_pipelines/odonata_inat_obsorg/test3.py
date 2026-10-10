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
#         #"thorax": [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"),mask_threshold=1.0)],
#         "head": [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"), mask_threshold=1.0)],
#     },
# )

cf.grow_subset(PROJECT_PATH, name="thorax_annotations", target_size=200)
cf.run_metrics(PROJECT_PATH, subset="thorax_annotations",
               part="thorax",
               transforms=[cf.remove_background(),
                           cf.remove_islands(),
                           cf.crop_to_mask(),
                           cf.orient(),
                           cf.erode(fraction=0.15)],
               metrics=[cf.exclusive_label_annotation(labels=["perfect",
                                                              "good",
                                                              "poor"],
                                                      note="Note that these are judgements about segment shape, NOT illumination"
                                                           "perfect: the part entire, not curved, not covered by wing, and with <5% background bleed"
                                                           "good: representative of color pattern of part but doesn't have to be entire part and can be curved - for example half of abdomen containing all the colors that exist across the whole abdomen approximately proportional to whole abdomen colors\n"
                                                           "poor: not representative of color pattern of part, not of part at all, has >5% background bleed \n",
                                                      show_original=True,
                                                            )])
cf.run_metrics(
    PROJECT_PATH, run_name="thorax_qc",
    part="thorax",
    metrics=[cf.mask_info(operations=["segment"]),
             cf.mask_fraction(), cf.blur_variance(),cf.mask_area(),cf.n_islands(),
             cf.elongation(), cf.jaggedness()],
)

filters = cf.get_validated_filters(
    PROJECT_PATH,
    part="thorax",
    metric_specs={
        "thorax_qc": {
            "mask_info__segment__score": "below", # mask not confident
            "blur_variance": "below",  # out of focus
            "mask_fraction": "below",  # tiny within image
            "mask_area": "below",  # tiny in general
            "n_islands": "above", # many fragments
            "elongation": "above", # too slender for a roughly round thorax
            "jaggedness": "above", # shouldn't be too jagged
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="thorax_annotations",
    max_fpr=0.1,
    min_precision=0.8,
    drop_redundant=True,     # a filter that catches nothing the others miss only costs good organisms
)

#
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="abdomen_color_thresholds",
#     part="abdomen",
#     transforms=[cf.remove_background(),
#                    cf.remove_islands(),
#                    cf.erode(fraction=0.15)],
#     metrics=[
#         cf.threshold_fractions([
#             # LCh hue of reference sRGB colours: red 36, orange-red 46, orange 64,
#             # amber 83, yellow 100, green 141, blue 297, violet 313, purple 321, magenta 339
#             cf.color_threshold("red",        lch_h=(345, 50),  lch_c=(30, None)),
#             cf.color_threshold("orange",     lch_h=(50, 75),   lch_c=(30, None)),
#             cf.color_threshold("yellow",     lch_h=(75, 105),  lch_c=(30, None)),
#             cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
#             cf.color_threshold("blue",       lch_h=(195, 305), lch_c=(30, None), lch_l=(None, 55)),
#             cf.color_threshold("light_blue", lch_h=(195, 305), lch_c=(20, None), lch_l=(55, None)),
#             cf.color_threshold("purple",     lch_h=(305, 345), lch_c=(30, None)),
#         ], unmatched=True, name="color_bins"),
#         # present at >= 5% of the abdomen; n_colors_present; ranked_color_1/2
#         cf.color_presence(min_fraction=0.05, n_ranked_colors=2),
#     ],
#     force=True
# )