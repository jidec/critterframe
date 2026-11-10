"""
A size lineup, as in a museum drawer: specimens oriented, drawn to scale by their
body length, smallest to largest, standing on a common baseline, with a scale bar.

scale_by draws each specimen with its longest side in proportion to its measured
length, so the cutouts are oriented the same way the length was measured. units="mm"
converts the length with the project's scale calibration, so the bar is in
millimetres; without one, leave units out and the bar is in pixels.

Before: a body_length run and a scale calibration (calibration/scale_calibration.py).
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

cf.run_metrics(
    PROJECT_PATH,
    run_name="body_dimensions",
    transforms=[cf.remove_appendages(), cf.orient()],
    metrics=[cf.body_length()],
)

cf.render_collage(
    PROJECT_PATH,
    "size_lineup",
    layout=cf.collage.grid(columns=12, anchor="bottom"),
    transforms=[cf.remove_appendages(), cf.orient(), cf.crop_to_mask(pad=0.02)],
    arrange=[
        cf.collage.sort_by("body_dimensions__organism__body_length_mm"),
        cf.collage.scale_by("body_dimensions__organism__body_length_mm"),
    ],
    units="mm",
    effects=[cf.collage.shadow()],
    decorations=[
        cf.collage.label("species"),
        cf.collage.scale_bar(10),
    ],
    background=(245, 245, 245),
)
