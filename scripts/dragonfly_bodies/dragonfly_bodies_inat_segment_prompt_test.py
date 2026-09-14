"""
Which text prompt should dragonfly_bodies_inat.py step 2's GroundedSAM2
detector use -- "dragonfly.", "insect.", or "subject."?

Grounding DINO's box detection is the one piece of that pipeline's whole-
organism segmentation that a text_prompt string decides, and which string
works best isn't obvious ahead of time: "dragonfly." is the most specific but
a detector may know the generic "insect." better, and "subject." is generic
enough to catch whatever's centered in the frame even when it doesn't
recognize the animal by name at all. This builds a small hand-drawn reference
set of organism masks once, then scores each candidate prompt's
detect_bounds=True chain against it by IoU -- the same held-out-set pattern
dragonfly_bodies_inat_training.py step 6 uses to score a trained part
segmenter, just with the candidate being a prompt string instead of a
checkpoint.

validate_masks(steps=...) computes each candidate's mask live and compares it
straight to the reference table, so nothing here writes a canonical organism
mask or competes with the main pipeline's own step 2 -- this only reads the
reference masks step 3 below writes. The winner is whatever step 2's
text_prompt= should read; this script doesn't write that decision back, since
a human should look at the IoU distribution and the diff panels first.

Run from the repo root:
    python scripts/dragonfly_bodies/dragonfly_bodies_inat_segment_prompt_test.py
"""

import logging

import critterframe as cf
from critterframe.project.subsets import select_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

REVIEW_SUBSET = "segment_prompt_review"
REFERENCE_SUBSET = "segment_prompt_reference"
SAMPLE_SIZE = 30

# Grounding DINO expects lowercase phrases ending in a period (see
# segmentation.groundedsam.DEFAULT_TEXT_PROMPT).
CANDIDATE_PROMPTS = ["dragonfly.", "insect.", "subject."]

# 1. A small sample to hand-draw reference organism masks over. Any subset
#    of occurrence ids works as the candidate pool -- sample_occurrences is
#    what makes the draw deterministic, not the pool it draws from.
cf.define_subset(
    PROJECT_PATH, name=REVIEW_SUBSET,
    occurrence_ids=cf.sample_occurrences(select_ids(PROJECT_PATH), count=SAMPLE_SIZE),
)

# 2. Screen before drawing -- same reasoning as body_parts_reference in
#    dragonfly_bodies_inat_training.py: a reference mask only means
#    something on a crop with one definable organism, not on debris, two
#    organisms, or one running off the frame edge.
cf.run_metrics(
    PROJECT_PATH, run_name="segment_prompt_screen", subset=REVIEW_SUBSET,
    metrics=[cf.annotate_flags()],
)

cf.define_subset(
    PROJECT_PATH, name=REFERENCE_SUBSET,
    occurrence_ids=cf.occurrences_matching(
        PROJECT_PATH, "segment_prompt_screen", {"annotate_flags": "usable"}),
)

# 3. Hand-draw the reference organism mask for each usable occurrence --
#    there's no upstream part to refine from, so draw_mask() rather than
#    correct_mask(). draw_mask() is deterministic=False (two people
#    painting one crop don't agree), so run_segments won't guess whether an
#    already-drawn occurrence should count as done or be redone;
#    force=False resumes a session left half-finished rather than redoing
#    everything drawn so far.
cf.run_segments(
    PROJECT_PATH,
    run_name="segment_prompt_reference",
    subset=REFERENCE_SUBSET,
    steps=[cf.draw_mask()],
    reference=True,
    force=False,
    visualize=False,
)

# 4. Score each candidate prompt's detect_bounds=True chain against that
#    reference set.
results = {}
for prompt in CANDIDATE_PROMPTS:
    logger.info("== text_prompt=%r ==", prompt)
    results[prompt] = cf.validate_masks(
        PROJECT_PATH,
        steps=[cf.segment(cf.groundedsam2(text_prompt=prompt))],
        visualize=True,
    )

ranked = sorted(
    results.items(),
    key=lambda item: item[1]["iou"].mean() if len(item[1]) else -1.0,
    reverse=True,
)
logger.info("== ranked by mean IoU ==")
for prompt, iou in ranked:
    logger.info("  %-12r mean=%.3f median=%.3f  (%d compared)", prompt,
                iou["iou"].mean() if len(iou) else float("nan"),
                iou["iou"].median() if len(iou) else float("nan"), len(iou))

logger.info("diff panels in %s/visualizations/validate_masks/", PROJECT_PATH)
