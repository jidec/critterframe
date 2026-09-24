"""
How good are the project's masks? Score them against the reference masks,
per occurrence.

Three questions, each one call:
1. The stored canonical masks against the references, as IoU.
2. The same, after holding a transform constant on both sides -- for when the
   references were drawn without appendages, say, so raw IoU would count every
   correctly omitted leg as an error.
3. A candidate recipe run live, without a run_segments() pass or writing
   anything -- for trying new settings before committing to them.

The population is every occurrence with a reference mask, so this measures
"agreement on the crops a human screened usable", not on everything.
Each call writes a worst-first diff grid and a score histogram under
visualizations/pipeline/ (white = agree, yellow = only predicted,
red = only reference).

Before: reference_annotation/ for the reference masks.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
PART = "organism"


def report(label, scores, column="iou"):
    logging.info("%-28s mean=%.3f median=%.3f min=%.3f (n=%d)", label,
                 scores[column].mean(), scores[column].median(),
                 scores[column].min(), len(scores))


# 1. Canonical masks as stored.
report("canonical", cf.validate_masks(PROJECT_PATH, part=PART, label="canonical"))

# 2. With appendages removed from both sides.
report("canonical, no appendages",
       cf.validate_masks(PROJECT_PATH, part=PART, transforms=[cf.remove_appendages()],
                         label="canonical_no_appendages"))

# 3. A candidate recipe, computed live. metric="coverage" instead of IoU when
#    extra predicted area shouldn't count, e.g. an organism mask that only has
#    to contain the body for a part segmenter downstream.
candidate_steps = [cf.segment(cf.groundedsam2(text_prompt="insect."), mask_threshold=-0.5)]
report("candidate mask_threshold=-0.5",
       cf.validate_masks(PROJECT_PATH, part=PART, steps=candidate_steps,
                         label="candidate_threshold_-0.5"))

# Several parts at once return {part: DataFrame}:
#   for part, scores in cf.validate_masks(PROJECT_PATH, parts=["head", "thorax"]).items():
#       report(part, scores)
