"""
The simplest possible CritterFrame pipeline, end to end.

Five calls: ingest a table of images, download them, segment the organism,
measure it, export. Everything else in the package is a variation on this.

This is also the AntWeb case as-is -- specimen images with a URL each,
whole-organism segmentation with GroundedSAM2, and colour metrics -- so start
here and add operations rather than starting from one of the more elaborate
scripts.

Needs an import CSV with an id column and an image-URL column. Point
PROJECT_PATH and IMPORT_CSV at real ones before running; nothing here has
defaults worth trusting.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/antweb_ants"
IMPORT_CSV = "imports/antweb_specimens.csv"

# 1. Ingest the occurrence table. Creates the project. The source CSV is
#    archived into the project's imports/ before anything is parsed, so this is
#    always recoverable.
cf.ingest_occurrences(
    PROJECT_PATH,
    IMPORT_CSV,
    id_col="specimen_id",
    image_url_col="image_url",
)

# 2. Download one image per occurrence into the project's image store. Only
#    fetches what's missing, so an interrupted download resumes.
cf.download_images(PROJECT_PATH)

# 3. Segment the whole organism. No part is named, so it defaults to
#    "organism". detect_bounds is on by default: these are specimen photographs
#    rather than pre-cut crops, so the organism has to be found before it can
#    be segmented.
cf.run_segments(
    PROJECT_PATH,
    steps=[
        cf.segment(cf.groundedsam2(text_prompt="ant.")),
    ],
)

# 4. Measure. Transforms clean up the working segment, metrics read values off
#    the end of it. Nothing here rewrites the stored mask -- what's recorded is
#    the recipe, so the measurement stays reproducible while the mask stays
#    the one thing every other measurement can also start from.
cf.run_metrics(
    PROJECT_PATH,
    run_name="traits",
    transforms=[
        cf.remove_appendages(),
    ],
    metrics=[
        cf.mean_lightness(),
        cf.mask_area(name="area_px", unit="px2"),
    ],
)

# 5. Export one row per occurrence. Columns are {run}__{part}__{metric}.
cf.export_metrics(PROJECT_PATH, f"{PROJECT_PATH}/traits.csv")

cf.print_summary(PROJECT_PATH)