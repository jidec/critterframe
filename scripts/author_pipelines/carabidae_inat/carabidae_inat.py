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
)
