"""
Step 8 of 8: render up to 1000 kept segments per part, for looking at what the
filters let through.

Reads the table 7_filter_and_export.py wrote. A part that failed its filters
has no value in any of its columns, so n_colors_present being filled in means
the part was kept. The abdomen is squeezed to a fixed strip; the head and
thorax keep their own proportions.

Unattended.

Before: 7_filter_and_export.py.
"""

import logging

import pandas as pd

import critterframe as cf
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# ids as text, so one like 007 isn't read as 7
df = pd.read_csv(exports_dir(PROJECT_PATH) / "part_colors.csv", dtype={"occurrence_id": str})

head_kept = df.loc[df["head_n_colors_present"].notna(), "occurrence_id"].tolist()[:1000]
cf.render_segments(PROJECT_PATH, name="head_renders_filtered", part="head",
                   occurrence_ids=head_kept,
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(), cf.erode(fraction=0.15), cf.crop_to_mask(),
                               cf.resize(height=100)])

thorax_kept = df.loc[df["thorax_n_colors_present"].notna(), "occurrence_id"].tolist()[:1000]
cf.render_segments(PROJECT_PATH, name="thorax_renders_filtered", part="thorax",
                   occurrence_ids=thorax_kept,
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(), cf.erode(fraction=0.15), cf.crop_to_mask(),
                               cf.resize(height=100)])

abdomen_kept = df.loc[df["abdomen_n_colors_present"].notna(), "occurrence_id"].tolist()[:1000]
cf.render_segments(PROJECT_PATH, name="abdomen_renders_filtered", part="abdomen",
                   occurrence_ids=abdomen_kept,
                   transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
                               cf.orient(axis_strategy="longer"), cf.erode(fraction=0.15),
                               cf.crop_to_mask(), cf.resize(20, 100)])
