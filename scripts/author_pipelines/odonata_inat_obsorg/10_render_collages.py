"""
Step 10 of 10: collages of the kept parts -- a hue rainbow, a colour wheel, an
average abdomen per species, a family-by-family tree, rings of exemplars and a
plate of dissected specimens.

Every collage draws only parts the filters kept, read from the table
7_filter_and_export.py wrote. A collage measures nothing, so the hue and chroma
it orders by come from a mean_color run first. That run covers only the
specimens drawn here (the "collage_specimens" subset), not the whole project.

Each render_collage call renders its own cutouts: transforms= is the cutout
chain, handed to render_segments under the render name "cutouts", so collages
sharing a chain share one folder of cutouts. Collages land in
visualizations/products/collages/ as <name>_<hash>.png, beside a .collage.json
and a .placements.csv. A rerun with nothing changed writes nothing.

Hue is noise on a brown or grey part, so the hue orderings are cleanest among
the bright ones; the colour wheel puts low-chroma parts near its center.

Unattended.

Before: 9_render_parts_species_exemplars.py.
"""

import logging

import pandas as pd

import critterframe as cf
from critterframe.project.paths import exports_dir

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
SEED = 0

# ids as text, so one like 007 isn't read as 7
df = pd.read_csv(exports_dir(PROJECT_PATH) / "part_colors.csv", dtype={"occurrence_id": str})
kept = {part: df[df[f"{part}_n_colors_present"].notna()] for part in ["head", "thorax", "abdomen"]}
all_kept = df[df[[f"{part}_n_colors_present" for part in kept]].notna().all(axis=1)]

# the most observed species, by kept abdomens
by_count = kept["abdomen"]["species"].value_counts().index.tolist()

# which specimens each collage draws
rainbow_ids = kept["abdomen"]["occurrence_id"].sample(2000, random_state=SEED).tolist()
wheel_ids = kept["thorax"]["occurrence_id"].sample(1500, random_state=SEED).tolist()
ghost_ids = (kept["abdomen"][kept["abdomen"]["species"].isin(by_count[:24])]
             .groupby("species").sample(50, random_state=SEED)["occurrence_id"].tolist())

# one per species, the most typical in colour proportions, as in 9_render_parts_species_exemplars.py
tree_ids = cf.exemplars_per_group(
    PROJECT_PATH, "color_thresholds", "species", part="abdomen", metric_name="color_bins",
    occurrence_ids=kept["abdomen"].loc[kept["abdomen"]["species"].isin(by_count[:200]), "occurrence_id"].tolist(),
    normalize=False)
ring_ids = cf.exemplars_per_group(
    PROJECT_PATH, "color_thresholds", "species", part="thorax", metric_name="color_bins",
    occurrence_ids=kept["thorax"].loc[kept["thorax"]["species"].isin(by_count[:300]), "occurrence_id"].tolist(),
    normalize=False)
# one occurrence per species whose head, thorax and abdomen were all kept
plate_ids = cf.exemplars_per_group(
    PROJECT_PATH, "color_thresholds", "species", part="abdomen", metric_name="color_bins",
    occurrence_ids=all_kept.loc[all_kept["species"].isin(by_count[:40]), "occurrence_id"].tolist(),
    normalize=False)

# 1. The colours collages order by, for these specimens only. Same pixels as color_thresholds.
cf.define_subset(PROJECT_PATH, "collage_specimens",
                 occurrence_ids=sorted(set(rainbow_ids + wheel_ids + ghost_ids + list(tree_ids)
                                           + list(ring_ids) + list(plate_ids))),
                 note="specimens drawn by 10_render_collages.py")
cf.run_metrics(PROJECT_PATH, run_name="mean_colors", parts=["head", "thorax", "abdomen"],
               subset="collage_specimens",
               transforms=[cf.remove_background(), cf.remove_islands(), cf.erode(fraction=0.15)],
               metrics=[cf.mean_color(name="lch", space="lch")])

# cutout chains, the same as 8_render_filtered_parts.py's
strip = [cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
         cf.orient(axis_strategy="longer"), cf.erode(fraction=0.15), cf.crop_to_mask(), cf.resize(20, 100)]
upright = [cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(),
           cf.orient(), cf.erode(fraction=0.15), cf.crop_to_mask(), cf.resize(height=100)]

# 2. Abdomens in rows, ordered by hue starting at red.
cf.render_collage(PROJECT_PATH, "abdomen_hue_rainbow", part="abdomen", occurrence_ids=rainbow_ids,
                  layout=cf.collage.grid(),
                  arrange=[cf.collage.sort_by("mean_colors__abdomen__lch__h", circular=True, start=0)],
                  transforms=strip, run_names=["mean_colors"], background=(0, 0, 0))

# 3. Thoraxes on a colour wheel: angle is hue, distance from the center is chroma, snapped so none overlap.
cf.render_collage(PROJECT_PATH, "thorax_color_wheel", part="thorax", occurrence_ids=wheel_ids,
                  layout=cf.collage.scatter(snap=True, density=1.5),
                  arrange=[cf.collage.polar_position("mean_colors__thorax__lch__h",
                                                     "mean_colors__thorax__lch__c")],
                  transforms=upright, run_names=["mean_colors"], background=(0, 0, 0))

# 4. The average abdomen of each of the 24 most observed species: 50 strips blended into one.
cf.render_collage(PROJECT_PATH, "abdomen_species_averages", part="abdomen", occurrence_ids=ghost_ids,
                  layout=cf.collage.nested(cf.collage.grid(columns=8), cf.collage.stack(), labels=True),
                  arrange=[cf.collage.group_by("species")],
                  transforms=strip, run_names=["mean_colors"], blend="mean", background=(255, 255, 255))

# 5. The 200 most observed species' abdomens as leaves of their family > genus > species tree.
cf.render_collage(PROJECT_PATH, "abdomen_taxonomy_tree", part="abdomen", occurrence_ids=tree_ids,
                  layout=cf.collage.dendrogram(radial=True),
                  arrange=[cf.collage.group_by(["family", "genus", "species"])],
                  transforms=strip, run_names=["mean_colors"], background=(255, 255, 255))

# 6. 300 species' thoraxes, a ring per family, ordered by hue within it, each turned to face outward.
cf.render_collage(PROJECT_PATH, "thorax_family_rings", part="thorax", occurrence_ids=ring_ids,
                  layout=cf.collage.rings(by_group=True, rotate="radial"),
                  arrange=[cf.collage.sort_by("mean_colors__thorax__lch__h", circular=True),
                           cf.collage.group_by("family", order="size")],
                  transforms=upright, run_names=["mean_colors"], effects=[cf.collage.shadow()],
                  background=(245, 245, 245))

# 7. A plate of 40 species, each one specimen's head, thorax and abdomen laid out top to bottom.
# One chain serves all three parts and isn't oriented, so each part keeps its pose in the photo.
cf.render_collage(PROJECT_PATH, "part_plates", parts=["head", "thorax", "abdomen"], occurrence_ids=plate_ids,
                  layout=cf.collage.nested(
                      cf.collage.grid(columns=8),
                      cf.collage.exploded({"head": (0, -1.1), "thorax": (0, 0), "abdomen": (0, 1.1)}),
                      labels=True),
                  arrange=[cf.collage.group_by("species")],
                  transforms=[cf.remove_background(), cf.remove_islands(), cf.crop_to_mask(pad=0.02)],
                  run_names=["mean_colors"], effects=[cf.collage.shadow()], background=(245, 245, 245))
