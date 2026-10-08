import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# one abdomen per species: the one whose colour proportions are most typical of the species.
# color_bins is the fractions 5_measure_part_qc_and_colors.py stored; already on one scale, so not normalized.
standard_ids = cf.exemplars_per_group(PROJECT_PATH, "abdomen_color_thresholds", "species",
                                      part="abdomen", metric_name="color_bins",
                                      normalize=True)
cf.render_segments(PROJECT_PATH, name="abdomen_standard_per_species", part="abdomen",
                   occurrence_ids=standard_ids, name_by="species",
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(axis_strategy="longer"), cf.erode(fraction=0.15),
                               cf.crop_to_mask(), cf.resize(20, 100)])