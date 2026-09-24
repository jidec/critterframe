"""
Step 3 of 3 of building a reference set: hand-correct masks on the usable
occurrences, into the reference mask table.

Reference masks live beside the canonical ones rather than replacing them, so
validation can measure the pipeline against them and a custom model can be
trained on them (custom_segmentation_model/). Each is started from an existing
mask and fixed by hand, which is several times faster than drawing from
scratch: left-drag erases, right-drag paints, '+'/'-' resize the brush, 's'
saves, Esc skips.

Before: annotate_usability.py, and a canonical `organism` segmentation to
correct.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

USABLE_SUBSET = "reference_candidates_usable"
MASK_REFERENCE_SUBSET = "mask_reference_set"

# Kept smaller than the usable set on purpose: masks cost minutes each, a
# screening label seconds. Grown from the usable subset's CURRENT members, so
# raising this later adds new specimens without touching finished ones.
TARGET_SIZE = 50

cf.grow_subset(PROJECT_PATH, MASK_REFERENCE_SUBSET, target_size=TARGET_SIZE,
               from_subset=USABLE_SUBSET)

# 1. Whole-organism reference masks, each starting from the canonical organism
#    mask. force=False is required rather than defaulted: two passes of a
#    hand-painted recipe hash alike but produce different masks, so the run
#    has to be told whether a rerun resumes (False) or redoes (True).
cf.run_segments(
    PROJECT_PATH,
    part="organism",
    from_part="organism",
    steps=[cf.correct_mask()],
    subset=MASK_REFERENCE_SUBSET,
    reference=True,
    force=False,
    batch_size=1,       # write each mask as it's saved, so quitting loses nothing
    visualize=False,
)

# 2. Optional: body-part reference masks, painted inside a crop of the organism.
#    The shared steps frame every part identically -- keep them the same as the
#    part-segmentation run they will validate. cf.draw_mask() instead of
#    cf.correct_mask() starts from an empty mask where no part mask exists yet.
PARTS = []   # e.g. ["head", "thorax", "abdomen"]
if PARTS:
    cf.run_segments(
        PROJECT_PATH,
        from_part="organism",
        shared_steps=[cf.remove_background(), cf.crop_to_mask(), cf.orient()],
        outputs={part: [cf.draw_mask()] for part in PARTS},
        subset=MASK_REFERENCE_SUBSET,
        reference=True,
        force=False,
        batch_size=1,
        visualize=False,
    )
