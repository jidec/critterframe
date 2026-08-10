"""
Salamanders in experimental boxes: local images, a specialized segmenter, and
position as the trait.

The loosest fit to the occurrence definition, and worth having as a reference
for exactly that reason. An occurrence here is one animal in one box in one
frame -- still one focal organism per image, so the model holds, but the
"place and time" is an experimental condition rather than a locality, and the
measurement of interest is WHERE the animal is rather than how big it is.

Three things differ from the other pipelines:

  Images are local files, not URLs, so ingest_images() does the whole job:
  images into the store and an occurrence row per file, keyed by filename.
  Filenames therefore have to be unique and stable -- renaming one later would
  orphan its mask and every metric measured from it.

  Segmentation is a specialized model only. A general text-prompted detector
  is the wrong tool for a fixed camera pointed at a fixed box: the setup never
  varies, so a small model trained on this exact scene beats a general one, and
  it's cheap to train because the reference masks come from the project itself.

  The metrics are positional. They report in ORIGINAL image coordinates,
  whatever transforms the recipe applied, which is what makes them comparable
  across frames -- a position in the coordinates of some intermediate crop
  would mean nothing.
"""

import logging

import critterframe as cf
from critterframe.metrics.position import centroid, image_bounds, relative_position

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/salamander_boxes"
FRAME_DIR = "data/box_frames"

# 1. Ingest the frames. metadata= joins per-frame experimental columns (box id,
#    treatment, timestamp) keyed by the filename stem, which is what the
#    occurrence ids are.
#
# import pandas as pd
# metadata = pd.read_csv("data/box_conditions.csv")   # occurrence_id, box, treatment, ...
cf.ingest_images(PROJECT_PATH, FRAME_DIR, recursive=True)

# 2. Build reference masks by hand for a training set, then train from them.
#    draw_mask() paints from scratch -- there's no model output to correct yet.
#    Scope this to a subset a person can actually get through.
#
# from critterframe.training.datasets import write_dataset
#
# cf.define_subset(PROJECT_PATH, "training_frames",
#                  occurrence_ids=["frame_0001", "frame_0002", ...])
# cf.run_segments(
#     PROJECT_PATH,
#     subset="training_frames",
#     run_name="hand_drawn",
#     steps=[cf.draw_mask()],
#     reference=True,      # into the reference table, beside the canonical one
# )
# write_dataset(PROJECT_PATH, "training/salamander", reference=True)

# 3. Segment with the trained model. Any object with
#    predict(image) -> (mask, score, info) and an identity() naming its
#    checkpoint works.
#
# from mymodels import salamander_segmenter
#
# cf.run_segments(
#     PROJECT_PATH,
#     steps=[cf.segment(salamander_segmenter())],
# )

# 4. Position, plus enough size to catch a bad mask. No orient() -- rotating
#    the frame would be actively wrong here, since the box's own geometry is
#    what position is measured against.
cf.run_metrics(
    PROJECT_PATH,
    run_name="position",
    metrics=[
        centroid(),
        relative_position(),
        image_bounds(),
        cf.mask_area(name="area_px", unit="px2"),
        cf.mask_fraction(),
    ],
)

# 5. Export with the experimental conditions alongside. mask_fraction filters
#    out frames where the segmenter found the whole box or a speck of nothing
#    -- either way, a position computed from that mask is meaningless.
#
#    Two bounds on ONE column, so this is a callable predicate rather than two
#    dict entries: a filters dict is keyed by column, and a second entry for
#    the same column would silently replace the first.
cf.export_metrics(
    PROJECT_PATH,
    f"{PROJECT_PATH}/positions.csv",
    occurrence_columns=["source_path", "source_mtime"],
    filters={
        "position__organism__mask_fraction": lambda s: s.between(0.001, 0.5),
    },
)

cf.print_summary(PROJECT_PATH)
