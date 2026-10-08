"""
Dragonfly bodies from iNaturalist observations published through GBIF

  1_ingest_download_and_segment_organism.py           unattended
  2_annotate_reference_masks_for_part_segmenters.py   manual by a person: usability labels, reference part masks
  3_train_part_segmenters.py                          training: head / thorax / abdomen segmenters
  4_segment_parts.py                                  unattended: head / thorax / abdomen
  5_measure_part_qc_and_colors.py                     unattended: part QC scores, colour traits
  6_annotate_parts_for_filters.py                     manual by a person: quality labels per part
  7_filter_and_export.py                              unattended: filters, export
  8_render_filtered_parts.py                          unattended: renders of the kept parts
"""

import logging

import critterframe as cf
from critterframe.extensions.gbif_darwincore_inat import ingest as gbif_ingest

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# ingest
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

# download
cf.download_images(PROJECT_PATH)

# foundational segmentation of organism using groundedsam2
# makes custom part segmentation task much easier by removing the background
cf.run_segments(
    PROJECT_PATH,
    run_name="organism_groundedsam2",
    steps=[
        cf.segment(cf.groundedsam2(text_prompt="insect.", box_threshold=0.25, text_threshold=0.25),
                   mask_threshold=0.0),
    ],
)