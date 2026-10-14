"""
Step 9 of 9: render one head, one thorax and one abdomen per species: the one
whose colour proportions are most typical of its species.

Chosen among the parts the filters kept, read from the table
7_filter_and_export.py wrote, by the colour proportions
5_measure_part_qc_and_colors.py stored. Each part is chosen on its own, so a
species' head, thorax and abdomen can come from three different occurrences.
Files are named by species.

Unattended.

Before: 8_render_filtered_parts.py.
"""

import logging

import pandas as pd

import critterframe as cf
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
  
# ids as text, so one like 007 isn't read as 7
df = pd.read_csv(exports_dir(PROJECT_PATH) / "part_colors.csv", dtype={"occurrence_id": str})

# color_bins is fractions, already on one scale, so not normalized.
head_kept = df.loc[df["head_n_colors_present"].notna(), "occurrence_id"].tolist()
head_exemplars = cf.exemplars_per_group(PROJECT_PATH, "color_thresholds", "species",
                                        part="head", metric_name="color_bins",
                                        occurrence_ids=head_kept, normalize=False)
cf.render_segments(PROJECT_PATH, name="head_exemplar_per_species", part="head",
                   occurrence_ids=head_exemplars, name_by="species",
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(), cf.erode(fraction=0.15), cf.crop_to_mask(),
                               cf.resize(height=100)])

thorax_kept = df.loc[df["thorax_n_colors_present"].notna(), "occurrence_id"].tolist()
thorax_exemplars = cf.exemplars_per_group(PROJECT_PATH, "color_thresholds", "species",
                                          part="thorax", metric_name="color_bins",
                                          occurrence_ids=thorax_kept, normalize=False)
cf.render_segments(PROJECT_PATH, name="thorax_exemplar_per_species", part="thorax",
                   occurrence_ids=thorax_exemplars, name_by="species",
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(), cf.erode(fraction=0.15), cf.crop_to_mask(),
                               cf.resize(height=100)])

abdomen_kept = df.loc[df["abdomen_n_colors_present"].notna(), "occurrence_id"].tolist()
abdomen_exemplars = cf.exemplars_per_group(PROJECT_PATH, "color_thresholds", "species",
                                           part="abdomen", metric_name="color_bins",
                                           occurrence_ids=abdomen_kept, normalize=False)
cf.render_segments(PROJECT_PATH, name="abdomen_exemplar_per_species", part="abdomen",
                   occurrence_ids=abdomen_exemplars, name_by="species",
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(axis_strategy="longer"), cf.erode(fraction=0.15),
                               cf.crop_to_mask(), cf.resize(20, 100)])
