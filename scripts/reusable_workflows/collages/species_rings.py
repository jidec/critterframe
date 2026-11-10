"""
Concentric rings, one species per ring, each specimen turned to face outward,
ordered by hue within its species.

group_by keeps each species together and keeps the hue order inside it, so the
order of the arrange steps reads innermost first: sort, then group. The spiral
beneath it is the same arrangement along one continuous curve.

Before: rainbow_grid.py, for the colors run.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

cf.render_collage(
    PROJECT_PATH,
    "species_rings",
    layout=cf.collage.rings(by_group=True, rotate="radial"),
    arrange=[
        cf.collage.sort_by("colors__organism__lch__h", circular=True),
        cf.collage.group_by("species"),
    ],
)

cf.render_collage(
    PROJECT_PATH,
    "hue_spiral",
    layout=cf.collage.spiral(rotate="tangent"),
    arrange=[cf.collage.sort_by("colors__organism__lch__h", circular=True)],
    background=(255, 255, 255),
)
