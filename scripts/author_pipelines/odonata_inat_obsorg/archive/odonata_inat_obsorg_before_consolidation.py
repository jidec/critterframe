"""
Dragonfly bodies from iNaturalist observations published through GBIF

The most elaborate of the reference pipelines, and the one that shows what the
package is actually for
"""

import logging

import critterframe as cf
from critterframe.metrics.outliers import outlier

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest
gbif_ingest.ingest_occurrences(
    PROJECT_PATH,
    archive_path="D:/0036785-260806074905277.zip",
    occurrence_columns=gbif_ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS,
    inat_photo_size="medium",
    dedupe_key_cols=["eventDate","decimalLatitude","decimalLongitude"],
    group_col="scientificName",
    max_per_group=500,
    trust_source_file_unchanged=True,
    prioritize_inat=True,
)

cf.download_images(PROJECT_PATH)

# #2. Whole organism. Persisted, and everything below starts from it.
# cf.run_segments(
#    PROJECT_PATH,
#    run_name="organism_groundedsam2",
#    steps=[
#        cf.segment(cf.groundedsam2(text_prompt="insect.",box_threshold=0.25, text_threshold=0.25),mask_threshold=0.0),
#    ],
# )

# Parts only for occurrences that pass the organism gate
# (odonata_inat_obsorg_organism_gate.py defines the subset).
# from critterframe.extensions.smp_segmenter import segmentation
# cf.run_segments(
#     PROJECT_PATH,
#     run_name="body_parts",
#     from_part="organism",
#     subset="organism_gate_pass",
#     shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
#     outputs={
#         "head":    [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"))],
#         "thorax":  [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"))],
#         "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
#     },
# )

#cf.run_segments(PROJECT_PATH, part="body", from_part=["head", "thorax", "abdomen"], steps=[],
#                subset="organism_gate_pass")

# newest run under this name (load_runs is newest first)
recipe_hash = cf.load_runs(PROJECT_PATH, kind="segment", name="body").iloc[0]["recipe_hash"]

# occurrences whose current mask for this part came from that recipe
done = cf.load_masks(PROJECT_PATH, parts=["body"], recipe_hash=recipe_hash,
                     columns=["occurrence_id", "part", "recipe_hash"])["occurrence_id"]

cf.define_subset(PROJECT_PATH, name="resnet_cluster_test_5000",
                 occurrence_ids=cf.sample_ids(done, 5000))


# # every bin also needs high chroma, so browns, greys and pruinose whites land in "unmatched".
# # light blue gets a lower chroma floor because pale blues are inherently low-chroma (sky blue ~26, brown ~24)
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="fixed_thresholds",
#     metrics=[
#         cf.threshold_fractions([
#             cf.color_threshold("red",        lch_h=(345, 55),  lch_c=(30, None)),
#             cf.color_threshold("yellow",     lch_h=(55, 105),  lch_c=(30, None)),
#             cf.color_threshold("green",      lch_h=(105, 195), lch_c=(30, None)),
#             cf.color_threshold("blue",       lch_h=(195, 320), lch_c=(30, None), lch_l=(None, 55)),
#             cf.color_threshold("light_blue", lch_h=(195, 320), lch_c=(20, None), lch_l=(55, None)),
#         ], unmatched=True, name="color_bins"),
#     ],
#     visualize=True,
# )
#
from critterframe.extensions.bioencoder.embedding import load_registered as load_bioencoder
from critterframe.metrics.embedding import pretrained

cf.run_metrics(
    PROJECT_PATH,
    subset="resnet_cluster_test_5000",
    part="abdomen",
    transforms=[cf.remove_background(), cf.remove_islands(),cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    metrics=[cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")],   # run_name defaults to "resnet18_embedding"
)

import critterframe.metrics.outliers as outliers
outliers.GALLERY_MEMBERS = 30  # images per cluster row

# cluster on resnet18 embeddings
cf.run_metrics(
    PROJECT_PATH,
    force=True,
    part="abdomen",
    subset="resnet_cluster_test_5000",
    run_name="resnet18_clusters_body_5000",
    metrics=[cf.cluster([cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")], from_run="resnet18_embedding",
                        n_clusters=5, n_components=8)],
)

#
# # # Trained on this project: the BioEncoder model odonata_inat_obsorg_bioencoder_training.py
# # # registered. Its weights are fingerprinted, so retraining re-embeds everything.
# # bioencoder_embedding = cf.embedding(load_bioencoder(PROJECT_PATH, MODEL_NAME),
# #                                     name="bioencoder_embedding")
# # cf.run_metrics(
# #     PROJECT_PATH,
# #     transforms=EMBEDDING_TRANSFORMS,
# #     metrics=[bioencoder_embedding],
# # )
#
# cf.run_metrics(
#     PROJECT_PATH,
#     metrics=[cf.mask_info()],
# )
#
# filters = cf.get_validated_filters(
#     PROJECT_PATH,
#     metric_specs=["edge_fraction", "bilateral_asymmetry","resnet18_embedding"],
#     predicted_run="qc",
#     annotation_run="human_annotation_labels",
#     max_fpr=0.20,   # "don't throw away more than X% of good data"
#     min_precision=0.5, # "at least X% of what I exclude should genuinely be bad"
# )
#
# cf.export_metrics(PROJECT_PATH)
