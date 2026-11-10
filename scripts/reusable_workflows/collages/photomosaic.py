"""
A photomosaic: a target image rebuilt from specimens, each grid cell filled by the
specimen whose mean colour is nearest the cell's.

mean_color() stores RGB on 0-1 scales, the same units mosaic() gives each cell, so
the three columns are matched directly. Specimens are reused across cells, and
tint() pulls each one a fifth of the way to its cell's colour.

Before: segmented organisms.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
TARGET_IMAGE = "figures/mosaic_target.jpg"

cf.run_metrics(
    PROJECT_PATH,
    run_name="rgb",
    transforms=[cf.remove_islands()],
    metrics=[cf.mean_color()],
)

cf.render_collage(
    PROJECT_PATH,
    "photomosaic",
    layout=cf.collage.mosaic(TARGET_IMAGE, columns=60),
    arrange=[
        cf.collage.match_on(
            [
                "rgb__organism__mean_color__r",
                "rgb__organism__mean_color__g",
                "rgb__organism__mean_color__b",
            ]
        )
    ],
    effects=[cf.collage.tint(0.2)],
    background=(0, 0, 0),
)
