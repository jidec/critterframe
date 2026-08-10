"""
Dragonfly bodies from iNaturalist: refinement chains, group metrics, and
embeddings.

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
from critterframe.extensions.inat_insects import download as inat_download
from critterframe.extensions.inat_insects import ingest as inat_ingest
from critterframe.extensions.inat_insects.metrics.color import (
    background_color,
    color_clusters,
    white_balanced_color,
)
from critterframe.metrics.outliers import outlier

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/dragonfly_bodies_inat"

# 1. Pull research-grade, openly-licensed observations. Rate-limited by the
#    API, so a large pull takes a while; it writes a CSV into the project's
#    imports/ first, so re-ingesting never means re-querying.
inat_ingest.ingest_occurrences(
    PROJECT_PATH,
    taxon_name="Odonata",
    place_id=1,          # United States
    limit=5000,
)
inat_download.download_images(PROJECT_PATH)

# 2. Whole organism. Persisted, and everything below starts from it.
cf.run_segments(
    PROJECT_PATH,
    run_name="organism_sam2",
    steps=[
        cf.segment(cf.groundedsam2(text_prompt="dragonfly.")),
    ],
)

# 3. Body parts, refined from the organism mask. Each part-specific model is
#    caller-supplied: anything with predict(image) -> (mask, score, info) and
#    an identity() naming its checkpoint. Train them from this project's own
#    reference masks -- see critterframe.training.datasets.
#
# from mymodels import head_segmenter, thorax_segmenter, abdomen_segmenter
#
# cf.run_segments(
#     PROJECT_PATH,
#     run_name="body_parts",
#     from_part="organism",
#     shared_steps=[
#         cf.remove_background(),
#         cf.crop_to_mask(),
#         cf.orient(),
#     ],
#     outputs={
#         "head":    [cf.segment(head_segmenter())],
#         "thorax":  [cf.segment(thorax_segmenter())],
#         "abdomen": [cf.segment(abdomen_segmenter())],
#     },
# )

# 4. Whole-organism traits and QC. The traits run is named separately from the
#    QC run so revising a QC threshold doesn't invalidate the traits.
cf.run_metrics(
    PROJECT_PATH,
    run_name="body_dimensions",
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

cf.run_metrics(
    PROJECT_PATH,
    run_name="qc",
    transforms=[cf.remove_appendages(), cf.orient()],
    metrics=[
        cf.blur_variance(),
        cf.bilateral_asymmetry(),
        cf.edge_fraction(),
        cf.mask_fraction(),
        # The background is context, not noise: what the animal was
        # photographed against, and how much it stands out from it -- which is
        # the single best predictor of whether the mask is any good.
        background_color(),
    ],
)

# 5. Outlier detection within each species. Fits against the body_dimensions
#    values already stored, then scores each occurrence from its own segment --
#    so an occurrence that wasn't in the reference population still gets a
#    score. Being an outlier WITHIN your own species is the QC-relevant signal;
#    comparing across species would mostly rediscover that species differ in
#    size.
cf.run_metrics(
    PROJECT_PATH,
    run_name="species_qc",
    transforms=[cf.remove_appendages(), cf.orient()],
    metrics=[
        outlier(
            features=[cf.body_length(), cf.max_width()],
            from_run="body_dimensions",
            group_col="taxon",
        ),
    ],
)

# 6. Colour. Per-part where the parts exist (head/thorax/abdomen differ in
#    colour and that's the point), whole-organism otherwise. The cluster metric
#    fits one palette per species and reports what fraction of each individual
#    falls in each colour -- a signature that captures pattern, which a mean
#    colour cannot.
COLOR_TRANSFORMS = [cf.remove_appendages(), cf.orient()]

cf.run_metrics(
    PROJECT_PATH,
    run_name="body_color",
    transforms=COLOR_TRANSFORMS,
    metrics=[
        white_balanced_color(),
        cf.mean_lightness(),
        cf.black_fraction(threshold=0.20),
        cf.red_fraction(),
        cf.yellow_fraction(),
        color_clusters(
            n_colors=5,
            group_col="taxon",
            # the same transforms the run uses, so the palette is fit on the
            # same representation it will be scoring
            transforms=COLOR_TRANSFORMS,
        ),
    ],
)

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

# 8. Export, dropping the occurrences flagged as outliers or as poor images.
cf.export_metrics(
    PROJECT_PATH,
    f"{PROJECT_PATH}/dragonfly_traits.csv",
    occurrence_columns=["taxon", "genus", "family", "observed_on",
                        "latitude", "longitude", "license_code", "observer"],
    filters={
        "species_qc__organism__outlier__is_outlier": ("==", False),
        "qc__organism__edge_fraction": ("<=", 0.02),
    },
)

cf.print_summary(PROJECT_PATH)
