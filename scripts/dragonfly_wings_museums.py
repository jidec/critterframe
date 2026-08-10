"""
Dragonfly wings from several museum collections: subsets and parts.

The case this exists to demonstrate: one analytical dataset whose members were
imaged to incompatible standards. Three collections photographed their
specimens differently -- wings laid out in different positions on the sheet --
but the wings themselves are the same structures and the traits are the same
traits, so this is ONE project, not three.

Two data-model decisions carry it:

  SUBSETS. Each collection is a named selection of occurrences receiving its
  own segmentation recipe. Running one subset's recipe leaves every other
  subset's masks untouched, so the three can be processed independently and
  still land in one table.

  PARTS. Four wings on one sheet are four PARTS of one occurrence, not four
  occurrences -- they belong to one organism. Each is reached by cropping to
  where it sits, and the mask is inverted back to the parent image's
  coordinates before it's stored, so all four end up describing pixels of the
  same photograph.

Metrics are then run once across all four parts and every subset, because by
that point the differences in how they were imaged have been absorbed.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/dragonfly_wings"
IMPORT_CSV = "imports/specimens.csv"

cf.ingest_occurrences(
    PROJECT_PATH,
    IMPORT_CSV,
    id_col="specimen_id",
    image_url_col="image_url",
)
cf.download_images(PROJECT_PATH)

# Define the subsets once, from the column that already separates them. This
# writes definitions/subsets.toml, which is hand-editable afterward -- knowing
# which collection is which is human knowledge about the data.
cf.define_subsets(
    PROJECT_PATH,
    column="collection",
    mapping={
        "Alabama Museum": "alabama",
        "May Collection": "may",
        "AMNH": "amnh",
    },
)

WINGS = ("left_forewing", "right_forewing", "left_hindwing", "right_hindwing")

# The Alabama sheets put one wing in each quadrant, so a named region is enough.
cf.run_segments(
    PROJECT_PATH,
    subset="alabama",
    run_name="wing_parts",
    outputs={
        "left_forewing":  [cf.crop(region="upper_left"),  cf.segment(cf.sam2())],
        "right_forewing": [cf.crop(region="upper_right"), cf.segment(cf.sam2())],
        "left_hindwing":  [cf.crop(region="lower_left"),  cf.segment(cf.sam2())],
        "right_hindwing": [cf.crop(region="lower_right"), cf.segment(cf.sam2())],
    },
)

# The AMNH sheets are laid out differently and mounted at an angle, so this
# subset gets explicit boxes and a rotation. Same parts, same downstream
# metrics, different recipe -- and a different recipe hash, so the two subsets'
# work is tracked separately and neither disturbs the other.
cf.run_segments(
    PROJECT_PATH,
    subset="amnh",
    run_name="wing_parts",
    outputs={
        "left_forewing": [
            cf.crop(x=120, y=80, width=900, height=620),
            cf.rotate(-12),
            cf.segment(cf.sam2()),
        ],
        "right_forewing": [
            cf.crop(x=1060, y=80, width=900, height=620),
            cf.rotate(12),
            cf.segment(cf.sam2()),
        ],
        "left_hindwing": [
            cf.crop(x=120, y=740, width=900, height=620),
            cf.segment(cf.sam2()),
        ],
        "right_hindwing": [
            cf.crop(x=1060, y=740, width=900, height=620),
            cf.segment(cf.sam2()),
        ],
    },
)

# One metric run across every part and every subset. The imaging differences
# are gone by now -- each mask is in its own image's coordinates, and area is
# area.
cf.run_metrics(
    PROJECT_PATH,
    run_name="wing_traits",
    parts=list(WINGS),
    transforms=[
        cf.orient(),
    ],
    metrics=[
        cf.mask_area(name="area_px", unit="px2"),
        cf.body_length(name="wing_length"),
        cf.max_width(name="wing_width"),
    ],
)

# One row per specimen, four wings' worth of columns each.
cf.export_metrics(
    PROJECT_PATH,
    f"{PROJECT_PATH}/wing_traits.csv",
    occurrence_columns=["collection", "taxon"],
)

cf.print_summary(PROJECT_PATH)
