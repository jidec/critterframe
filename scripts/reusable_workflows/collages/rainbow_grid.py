"""
A rainbow of specimens: every organism cut out and laid in rows, ordered by hue.

A collage measures nothing, so the hue comes from a colour run first. mean_color in
LCh space stores lightness, chroma and hue per specimen, the hue as an angle in
degrees, which is why the sort is circular: a red at 355 degrees sits beside a red
at 5. The run is skipped for every specimen it already measured.

The collage lands in visualizations/products/collages/ as <name>_<hash>.png, with
a .collage.json saying which run, recipe and specimens it shows and a
.placements.csv saying who sits where. A rerun with nothing changed writes nothing.

Before: segmented organisms (simplest_full_pipeline.py).
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

# 1. The colour each specimen is ordered by.
cf.run_metrics(
    PROJECT_PATH,
    run_name="colors",
    transforms=[cf.remove_islands()],
    metrics=[cf.mean_color(name="lch", space="lch")],
)

# 2. The collage: a grid, neighbors overlapping by a tenth, hue order starting at red.
cf.render_collage(
    PROJECT_PATH,
    "rainbow_grid",
    layout=cf.collage.grid(overlap=0.1),
    arrange=[cf.collage.sort_by("colors__organism__lch__h", circular=True, start=0)],
    background=(0, 0, 0),
)
