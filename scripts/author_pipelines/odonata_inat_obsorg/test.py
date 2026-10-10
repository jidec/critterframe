import logging

import critterframe as cf
from critterframe.selection.subsets import select_ids
from critterframe.selection.queries import ids_with_mask
from critterframe.metrics.embedding import pretrained
from critterframe.extensions.smp_segmenter import segmentation

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# cf.run_metrics(
#     PROJECT_PATH, part="organism",
#     transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
#                 cf.orient(axis_strategy="longer")],
#     metrics=[cf.embedding(pretrained("resnet18", resize="pad"),
#                                 name="resnet18_embedding")],   # run_name defaults to "resnet18_embedding"
# )
#
# cf.grow_subset(PROJECT_PATH, name="organism_annotations", target_size=200)
# cf.run_metrics(PROJECT_PATH, subset="organism_annotations",
#                metrics=[cf.exclusive_label_annotation(labels=["perfect_dorsal","perfect_dorsoventral","perfect_ventral",
#                                                               "ok_dorsal","ok_dorsoventral","ok_ventral",
#                                                               "poor"],
#                                                       note="dorsal: entire dorsal side color pattern perceptible with nothing occluded by angle or wings/objects\n"
#                                                            "dorsoventral: angled within 30 deg of the midpoint between dorsal and ventral\n"
#                                                            "ventral: entire ventral side color pattern perceptible\n"
#                                                             "perfect criterion: no occlusion, not blurry, lighting makes color pattern clearly visible\n"
#                                                             "ok criterion: 0-20% occluded (often by damselfly wings), somewhat blurry or somewhat poorly lit such that pattern is slightly obscured"
#                                                            "poor: failed segment, >20% occluded, very blurry or very poorly lit, angled too far front or back such that pattern is lost, multiple orgs, wrong life stage"
#                                                             )])
#
# cf.run_metrics(
#     PROJECT_PATH, run_name="organism_qc", subset="organism_annotations",
#     metrics=[cf.mask_info(operations=["segment"]), cf.edge_fraction(),
#              cf.mask_fraction(), cf.blur_variance(), cf.mask_area()],
# )
#
# # Outliers and clusters among the resnet embeddings. Each model is fitted on,
# # and scores, a 10,000-organism reference sample plus every annotated organism,
# # so the annotated ones all have a value to calibrate against. Organisms outside
# # this subset have no outlier score or cluster yet.
# cf.define_subset(
#     PROJECT_PATH, name="organism_outlier_reference",
#     occurrence_ids=sorted(set(cf.sample_ids(select_ids(PROJECT_PATH), 10000))
#                           | set(select_ids(PROJECT_PATH, subset="organism_annotations"))),
# )
#
# resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
#                                 name="resnet18_embedding")
#
# cf.run_metrics(
#     PROJECT_PATH, run_name="organism_outliers", part="organism",
#     subset="organism_outlier_reference",
#     metrics=[cf.outlier([resnet_embedding], from_run="resnet18_embedding")],
# )
#
# cf.run_metrics(
#     PROJECT_PATH, run_name="organism_clusters", part="organism",
#     subset="organism_outlier_reference",
#     metrics=[cf.cluster([resnet_embedding], from_run="resnet18_embedding",
#                         n_clusters=7, n_components=8)],
# )

gate = cf.get_validated_filters(
    PROJECT_PATH,
    metric_specs={
        "organism_qc": {
            "mask_info__segment__box_score": "below",          # detector unsure
            #"mask_info__segment__score": "below",              # SAM2 unsure of its mask
            "mask_info__segment__second_box_score": "above",   # a second organism in frame
            "edge_fraction": "above",                          # mask runs off the frame
            "blur_variance": "below",                          # out of focus
            "mask_fraction": "below",                          # tiny within image
            "mask_area": "below",                              # tiny in general
        },
        "organism_outliers": {
            # IsolationForest's decision function: lower is more anomalous
            "outlier__anomaly_score": "below",
        },
        "organism_clusters": {
            # the worst clusters are dropped, within the same constraint; a
            # cluster with fewer than 10 labels is never dropped
            "cluster__cluster_id": "category",
        },
    },
    bad_labels=["poor"],
    annotation_run="exclusive_label_annotation",
    label_metric="exclusive_label_annotation",
    subset="organism_annotations",
    max_fpr=0.1,
    min_precision=0.5,
    drop_redundant=True,     # a filter that catches nothing the others miss only costs good organisms
)

cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
    },
)