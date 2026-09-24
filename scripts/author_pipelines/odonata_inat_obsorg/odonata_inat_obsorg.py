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

#2. Whole organism. Persisted, and everything below starts from it.
cf.run_segments(
   PROJECT_PATH,
   run_name="organism_groundedsam2",
   steps=[
       cf.segment(cf.groundedsam2(text_prompt="insect.",box_threshold=0.25, text_threshold=0.25),mask_threshold=0.0),
   ],
)

from critterframe.extensions.smp_segmenter import segmentation
cf.run_segments(
    PROJECT_PATH,
    run_name="body_parts",
    from_part="organism",
    shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    outputs={
        "head":    [cf.segment(segmentation.load_registered(PROJECT_PATH, "head_segmenter_v1"))],
        "thorax":  [cf.segment(segmentation.load_registered(PROJECT_PATH, "thorax_segmenter_v1"))],
        "abdomen": [cf.segment(segmentation.load_registered(PROJECT_PATH, "abdomen_segmenter_v1"))],
    },
)

from critterframe.metrics.inductive_color_thresholds import inductive_color_thresholds
from critterframe.records.masks import occurrence_ids_with_mask
from critterframe.selectionhelpers import sample_occurrences

ids = sample_occurrences(occurrence_ids_with_mask(PROJECT_PATH,"head"),2500)
cf.define_subset(PROJECT_PATH,name="thresholds_test",
                 occurrence_ids=ids)

# every bin also needs high chroma, so browns, greys and pruinose whites land in "unmatched".
# light blue gets a lower chroma floor because pale blues are inherently low-chroma (sky blue ~26, brown ~24)
MIN_CHROMA = 30
LIGHT_BLUE_MIN_CHROMA = 20
cf.run_metrics(
    PROJECT_PATH,
    run_name="fixed_thresholds",
    subset="thresholds_test",
    metrics=[
        cf.threshold_fractions([
            cf.color_threshold("red",        lch_h=(345, 55),  lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("yellow",     lch_h=(55, 105),  lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("green",      lch_h=(105, 195), lch_c=(MIN_CHROMA, None)),
            cf.color_threshold("blue",       lch_h=(195, 320), lch_c=(MIN_CHROMA, None), lch_l=(None, 55)),
            cf.color_threshold("light_blue", lch_h=(195, 320), lch_c=(LIGHT_BLUE_MIN_CHROMA, None), lch_l=(55, None)),
        ], unmatched=True, name="color_bins"),
    ],
    visualize=True,
)

#3. Embeddings. One vector per organism, stored as a metric like any trait.
# The same transform chain the BioEncoder model was trained on, imported rather
# than retyped: a model run on differently-prepared segments embeds badly with
# no error to say so. Importable because that script's work sits under __main__.
from critterframe.extensions.bioencoder.embedding import load_registered as load_bioencoder
from critterframe.metrics.embedding import pretrained
from odonata_inat_obsorg_bioencoder_training import EMBEDDING_TRANSFORMS, MODEL_NAME

# Off-the-shelf: an ImageNet resnet18 from timm, classifier removed. resize="pad"
# letterboxes so a long, thin abdomen isn't squashed into a square.
resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")
cf.run_metrics(
    PROJECT_PATH,
    transforms=EMBEDDING_TRANSFORMS,
    metrics=[resnet_embedding],   # run_name defaults to "resnet18_embedding"
)

# Trained on this project: the BioEncoder model odonata_inat_obsorg_bioencoder_training.py
# registered. Its weights are fingerprinted, so retraining re-embeds everything.
bioencoder_embedding = cf.embedding(load_bioencoder(PROJECT_PATH, MODEL_NAME),
                                    name="bioencoder_embedding")
cf.run_metrics(
    PROJECT_PATH,
    transforms=EMBEDDING_TRANSFORMS,
    metrics=[bioencoder_embedding],
)

#4. Cluster the stored embeddings. Reads the vectors the runs above stored --
# no image or network is touched -- so no transforms are passed. The features
# must be the SAME operations those runs measured with, hence reusing them.
# Within each species (group_col), so clusters are colour morphs, sexes, or bad
# masks rather than a rediscovery of species; a species with fewer than 20
# embedded specimens shares the population-wide fit. Each run draws a gallery
# per species -- a row of sampled segments per cluster -- under pipeline/.
cf.run_metrics(
    PROJECT_PATH,
    run_name="resnet18_clusters",
    metrics=[cf.cluster([resnet_embedding], from_run="resnet18_embedding",
                        group_col="species", n_clusters=3, n_components=16,
                        min_group_size=20)],
)