"""
Dragonfly bodies from iNaturalist observations published through GBIF:
refinement chains, group metrics, and embeddings.

Ingest reads a GBIF Darwin Core Archive rather than calling the iNaturalist API
directly -- gbif_darwincore_inat works for any GBIF-published occurrence data,
and iNaturalist's research-grade observations happen to be the dataset this
script targets. See extensions/gbif_darwincore_inat for the archive shape and
extensions/inat_insects for the live-API alternative.

The source archive is a combined Odonata pull covering more than one
GBIF-mediated aggregator (iNaturalist and Observation.org both feed GBIF), but
this project keeps only the iNaturalist-published rows (_inaturalist_only,
below) -- simpler than cross-source deduplication
(gbif_ingest.dedupe_key_cols) when a single source's photos are all a project
actually needs.

The most elaborate of the reference pipelines, and the one that shows what the
package is actually for:

  REFINEMENT. Whole-organism segmentation runs first and is persisted, then
  body-part segmentation starts FROM that mask (from_part="organism") rather
  than rediscovering the animal. The organism mask is worth persisting on all
  three counts -- it's a real part, everything downstream depends on it, and
  producing it is expensive.

  SHARED PREPROCESSING. Head, thorax, and abdomen share their background
  removal and orientation; those run once per occurrence and the segment forks
  per part, so three parts cost one preprocessing pass rather than three.

  GROUP METRICS. Outlier detection and colour clustering can't score an
  occurrence in isolation -- they need to know what the rest of the species
  looks like first. They're still metrics: they compose into a recipe and store
  like anything else, and the fitting happens once before the run's loop.

  EMBEDDINGS. A learned vector per organism, stored beside the hand-designed
  traits, for the differences nobody has written a measurement for.

iNaturalist photographs are uncontrolled, so QC matters more here than in a
specimen-imaging project, and absolute size is not recoverable at all -- there
is no reference object, so every trait is in pixels and comparable only as a
ratio.
"""

import logging

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest
from critterframe.metrics.outliers import outlier

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# 1. Normalize the archive's occurrence.txt + multimedia.txt into one table,
#    rewriting each selected photo's URL to the full-resolution rendition --
#    GBIF's own iNaturalist photos are re-hosted on S3, not iNaturalist's API,
#    so nothing here is rate-limited the way a live iNaturalist pull is.
gbif_ingest.ingest_occurrences(
    PROJECT_PATH,
    archive_path="D:/0036785-260806074905277.zip",
    occurrence_columns=gbif_ingest.DEFAULT_INAT_OCCURRENCE_COLUMNS,
    inat_photo_size="medium",
    dedupe_key_cols=["eventDate","decimalLatitude","decimalLongitude"],
    group_col="scientificName",
    max_per_group=500,
    trust_source_file_unchanged=True,
)

cf.download_images(PROJECT_PATH)

#2. Whole organism. Persisted, and everything below starts from it.
cf.run_segments(
   PROJECT_PATH,
   run_name="organism_sam2",
   steps=[
       cf.segment(cf.groundedsam2(text_prompt="dragonfly.")),
   ],
)

# 3. Body parts, refined from the organism mask, using the UNet++ segmenters
#    trained -- and IoU-checked against a held-out test split -- in
#    dragonfly_bodies_inat_training.py's steps 5/6. load_part_segmenter()
#    reads each one out of the model registry (models/registry.json), so
#    retraining and re-registering a checkpoint under the same name is all a
#    rerun of this script needs to pick up the new weights.
#
#    shared_steps MUST match BODY_PART_TRANSFORMS from that same script
#    exactly: the whole reason its dataset export used from_part="organism"
#    was to give each model this identical shared crop at inference, not one
#    cropped to its own, usually much smaller, mask. Importing the constant
#    (rather than repeating the three transforms here) is what keeps the two
#    from silently drifting apart.
from scripts.dragonfly_bodies_inat_training import (
    BODY_PART_TRANSFORMS, load_part_segmenter,
)

cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    shared_steps=BODY_PART_TRANSFORMS,
    outputs={
        "head":    [cf.segment(load_part_segmenter("head_segmenter_v1"))],
        "thorax":  [cf.segment(load_part_segmenter("thorax_segmenter_v1"))],
        "abdomen": [cf.segment(load_part_segmenter("abdomen_segmenter_v1"))],
    },
)

# 4. Whole-organism traits and QC. The traits run is named separately from the
#    QC run so revising a QC threshold doesn't invalidate the traits.
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="body_dimensions",
#     transforms=[
#         cf.remove_appendages(),
#         cf.orient(),
#     ],
#     metrics=[
#         cf.body_length(),
#         cf.max_width(),
#         cf.mask_area(name="area_px", unit="px2"),
#     ],
# )
#
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="qc",
#     transforms=[cf.remove_appendages(), cf.orient()],
#     metrics=[
#         cf.blur_variance(),
#         cf.bilateral_asymmetry(),
#         cf.edge_fraction(),
#         cf.mask_fraction(),
#         # The background is context, not noise: what the animal was
#         # photographed against, and how much it stands out from it -- which is
#         # the single best predictor of whether the mask is any good.
#         background_color(),
#     ],
#)

# 5. Outlier detection within each species. Fits against the body_dimensions
#    values already stored, then scores each occurrence from its own segment --
#    so an occurrence that wasn't in the reference population still gets a
#    score. Being an outlier WITHIN your own species is the QC-relevant signal;
#    comparing across species would mostly rediscover that species differ in
#    size.
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="species_qc",
#     transforms=[cf.remove_appendages(), cf.orient()],
#     metrics=[
#         outlier(
#             features=[cf.body_length(), cf.max_width()],
#             from_run="body_dimensions",
#             group_col="species",
#         ),
#     ],
# )

# 6. Colour. Per-part where the parts exist (head/thorax/abdomen differ in
#    colour and that's the point), whole-organism otherwise. The cluster metric
#    fits one palette per species and reports what fraction of each individual
#    falls in each colour -- a signature that captures pattern, which a mean
#    colour cannot.
# COLOR_TRANSFORMS = [cf.remove_appendages(), cf.orient()]
#
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="body_color",
#     transforms=COLOR_TRANSFORMS,
#     metrics=[
#         white_balanced_color(),
#         cf.mean_lightness(),
#         cf.black_fraction(threshold=0.20),
#         cf.red_fraction(),
#         cf.yellow_fraction(),
#         color_clusters(
#             n_colors=5,
#             group_col="species",
#             # the same transforms the run uses, so the palette is fit on the
#             # same representation it will be scoring
#             transforms=COLOR_TRANSFORMS,
#         ),
#     ],
# )

# 7. Embeddings, once a model has been trained. Expensive, and therefore the
#    clearest case for recipe hashing: rerun this and nothing is recomputed;
#    change the checkpoint and everything is, because the checkpoint is in the
#    hash.
#
# from critterframe.extensions.inat_insects.metrics.bioencoder import (
#     BioEncoderModel, embedding,
# )
#
# cf.run_metrics(
#     PROJECT_PATH,
#     run_name="embeddings",
#     transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient()],
#     metrics=[embedding(BioEncoderModel(my_network, "checkpoints/odonata_v1.pt"))],
# )

# # 8. Export, dropping the occurrences flagged as outliers or as poor images.
# cf.export_metrics(
#     PROJECT_PATH,
#     f"{PROJECT_PATH}/dragonfly_traits.csv",
#     occurrence_columns=["species", "genus", "family", "eventDate",
#                         "decimalLatitude", "decimalLongitude", "license",
#                         "recordedBy"],
#     filters={
#         "species_qc__organism__outlier__is_outlier": ("==", False),
#         "qc__organism__edge_fraction": ("<=", 0.02),
#     },
# )
#
# cf.print_summary(PROJECT_PATH)
