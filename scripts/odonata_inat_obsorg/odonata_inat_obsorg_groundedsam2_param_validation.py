"""
Which GroundedSAM2 settings work best for dragonfly_bodies_inat.py step 2 --
text_prompt, box_threshold, text_threshold, and segment()'s mask_threshold?

Each parameter is swept one at a time against a fixed baseline, ranked by how
much of the body (the union of the head/thorax/abdomen reference masks) the
candidate organism mask covers -- not IoU, since the organism mask feeds the
part segmenter and picking up the wings alongside the body costs nothing
there, only missing part of the body does.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
BODY_PARTS = ["head", "thorax", "abdomen"]

# text_prompt/box_threshold/text_threshold configure GroundedSAM2 itself;
# mask_threshold is segment()'s own param, applied to whatever model runs --
# kept in a separate baseline/sweep pair so it's never confused with a model
# kwarg.
MODEL_BASELINE = dict(text_prompt="insect.", box_threshold=0.25, text_threshold=0.25)
SEGMENT_BASELINE = dict(mask_threshold=0.0)

MODEL_SWEEPS = {
    "text_prompt": ["dragonfly.", "insect.", "subject."],
    "box_threshold": [0.15, 0.25, 0.35],
    "text_threshold": [0.15, 0.25, 0.35],
}
SEGMENT_SWEEPS = {
    "mask_threshold": [-1.0, -0.5, 0.0, 0.5, 1.0],
}

# The body the organism mask needs to cover, materialized once as a reference
# part so every sweep candidate below compares against it directly instead of
# re-unioning head/thorax/abdomen per candidate.
cf.merge_masks(PROJECT_PATH, parts=BODY_PARTS, into_part="body", reference=True)

def _validate(model_kwargs, segment_kwargs, label):
    return cf.validate_masks(
        PROJECT_PATH,
        part="organism",
        reference_part="body",
        metric="coverage",
        steps=[cf.segment(cf.groundedsam2(**model_kwargs), **segment_kwargs)],
        visualize=True,
        label=label,
    )


def _rank(results):
    return sorted(
        results.items(),
        key=lambda item: item[1]["coverage"].mean() if len(item[1]) else -1.0,
        reverse=True,
    )


def _log_ranked(param, results):
    logger.info("-- %s --", param)
    for value, scores in _rank(results):
        logger.info("  %-12r mean=%.3f median=%.3f  (%d compared)", value,
                    scores["coverage"].mean() if len(scores) else float("nan"),
                    scores["coverage"].median() if len(scores) else float("nan"),
                    len(scores))

for param, candidates in SEGMENT_SWEEPS.items():
    results = {value: _validate(MODEL_BASELINE, {**SEGMENT_BASELINE, param: value},
                                label=f"{param}={value}")
              for value in candidates}
    _log_ranked(param, results)

for param, candidates in MODEL_SWEEPS.items():
    results = {value: _validate({**MODEL_BASELINE, param: value}, SEGMENT_BASELINE,
                                label=f"{param}={value}")
              for value in candidates}
    _log_ranked(param, results)

# INFO:__main__:-- text_prompt --
# INFO:__main__:  'dragonfly.' mean=0.996 median=0.998  (82 compared)
# INFO:__main__:  'insect.'    mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  'subject.'   mean=0.982 median=0.998  (82 compared)
#
# INFO:__main__:-- box_threshold --
# INFO:__main__:  0.15         mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  0.25         mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  0.35         mean=0.996 median=0.997  (82 compared)
#
# INFO:__main__:-- text_threshold --
# INFO:__main__:  0.15         mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  0.25         mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  0.35         mean=0.996 median=0.997  (82 compared)
#
# INFO:__main__:-- mask_threshold --
# INFO:__main__:  -1.0         mean=0.998 median=0.999  (82 compared)
# INFO:__main__:  -0.5         mean=0.997 median=0.999  (82 compared)
# INFO:__main__:  0.0          mean=0.996 median=0.997  (82 compared)
# INFO:__main__:  0.5          mean=0.972 median=0.978  (82 compared)
# INFO:__main__:  1.0          mean=0.945 median=0.958  (82 compared)