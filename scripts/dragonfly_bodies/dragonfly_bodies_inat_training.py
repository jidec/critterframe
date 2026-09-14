"""
Trains dragonfly_bodies_inat.py's head/thorax/abdomen segmenters from this
project's own hand-drawn reference masks.

Run directly (`python -m scripts.dragonfly_bodies_inat_training`) to build the
reference set, export a dataset, and train+register a model per part. Also
IMPORTABLE for BODY_PART_TRANSFORMS and load_part_segmenter() -- what
dragonfly_bodies_inat.py's step 3 needs to run the registered models -- which
is why everything below is guarded by `if __name__ == "__main__":`: importing
this module must not re-run a training pass (or worse, block on
annotate_flags()'s human review) every time the main pipeline starts.
"""

import logging

import critterframe as cf
from critterframe.extensions.smp_segmenter import segmentation, training
from critterframe.project.subsets import select_ids, select_occurrences
from critterframe.records.masks import load_masks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
DATASET_DIR = f"{PROJECT_PATH}/training/body_parts"

PARTS = ["head", "thorax", "abdomen"]
BODY_PART_TRANSFORMS = [cf.remove_background(), cf.crop_to_mask(), cf.orient()]

# Occurrence ids to leave out of the exported training set entirely -- a bad
# drawing, a mislabeled specimen, whatever a look through the reference masks
# turns up. Populated by hand; empty means nothing is excluded yet.
EXCLUDED_REFERENCE_IDS = []

def load_part_segmenter(name):
    """
    A registered part segmenter, ready for run_segments -- what
    scripts/dragonfly_bodies_inat.py's step 3 uses in place of a
    "from mymodels import head_segmenter, thorax_segmenter, abdomen_segmenter"
    placeholder.
    """
    registered = cf.load_model(PROJECT_PATH, name)
    return registered.attach(segmentation.smp_segmenter(checkpoint=registered.path))


if __name__ == "__main__":
    reference_pool = select_occurrences(PROJECT_PATH, columns=["occurrence_id", "species"])
    masked = load_masks(PROJECT_PATH, parts=["organism"], columns=["occurrence_id","part"])  # part= as needed
    reference_pool = reference_pool[reference_pool["occurrence_id"].isin(masked["occurrence_id"])]

    cf.define_subset(
        PROJECT_PATH, name="body_parts_review",
        occurrence_ids=cf.sample_occurrences(reference_pool["occurrence_id"], count=100),
    )

    # 2. Screen the sample before drawing on it -- draw_mask() only makes sense on
    #    a crop with a definable head/thorax/abdomen, so flag occurrences with two
    #    organisms, no organism, or one running off the frame edge and exclude
    #    them below. Screen the WHOLE sample, including the ones that turn out
    #    unusable -- same reasoning as antenna_moths_pipeline.py's screening pass.
    cf.run_metrics(
        PROJECT_PATH, run_name="body_parts_screen", subset="body_parts_review",
        metrics=[cf.annotate_flags()],
    )

    # The crops a reference is even definable for -- defined by the HUMAN flag,
    # never by whether anything automated passed them. Same "reference" subset
    # pattern as antenna_moths_pipeline.py: only occurrences a person called
    # usable feed the hand-correction pass below, and therefore the training set
    # built from it.
    cf.define_subset(
        PROJECT_PATH, name="body_parts_reference",
        occurrence_ids=cf.occurrences_matching(
            PROJECT_PATH, "body_parts_screen", {"annotate_flags": "usable"}),
    )

    # 3. Hand-draw head/thorax/abdomen out of the organism crop, one occurrence at
    #    a time. Preprocessing runs ONCE per occurrence and the segment forks into
    #    three drawing windows -- the same shared-steps-then-fork shape every
    #    multi-part run uses, so one background removal costs one pass rather than
    #    three.
    #
    #    draw_mask(), not correct_mask(): there is no existing head/thorax/abdomen
    #    mask here to correct, only the ORGANISM mask feeding from_part, and
    #    correct_mask() would show that full silhouette as the starting mask -- so
    #    cancelling (Esc) would save the whole organism as e.g. "head". draw_mask()
    #    starts every window empty, so Esc instead raises "no mask was drawn",
    #    which run_segments logs and counts as failed rather than writing a wrong
    #    mask -- exactly "only export masks that weren't cancelled", enforced at
    #    the source rather than filtered out later. Left-drag erases, right-drag
    #    paints in, 's' saves.
    #
    #    force=False resumes a session left half-finished: draw_mask() is
    #    deterministic=False (two people drawing one crop don't agree), so
    #    run_segments refuses to guess whether an already-drawn occurrence should
    #    count as done or be redone -- force=True instead would redo every
    #    drawing made so far.
    # cf.run_segments(
    #     PROJECT_PATH,
    #     run_name="body_parts_reference",
    #     from_part="organism",
    #     subset="body_parts_reference",
    #     shared_steps=BODY_PART_TRANSFORMS,
    #     outputs={part: [cf.draw_mask()] for part in PARTS},
    #     reference=True,
    #     force=False,
    #     visualize=False,
    # )

    # 4. Export a train/test dataset per part from those reference masks, using
    #    the SAME transforms as the manual pass above.
    #
    #    A separate subset, not a mutation of body_parts_reference itself, so
    #    re-running step 2's screening still regenerates the same reference set
    #    and excluding a specimen from training stays a decision made once here
    #    rather than baked into the screening pass.
    cf.define_subset(
        PROJECT_PATH, name="body_parts_training",
        occurrence_ids=[occurrence_id for occurrence_id in
                        select_ids(PROJECT_PATH, subset="body_parts_reference")
                        if occurrence_id not in set(EXCLUDED_REFERENCE_IDS)],
    )

    #    from_part="organism" matters here as much as it did in step 3: without
    #    it, the exported image would be cropped/oriented to each part's OWN
    #    (small) mask instead of the shared organism crop the part-specific model
    #    will actually see at inference (step 6/main pipeline below) -- the image
    #    folder holds the unedited organism segment, not a part-specific one, with
    #    only the accompanying mask narrowed to that part.
    #    fractions has no "val" key at all, not val=0.0 -- split_ids() makes every
    #    key of the proportions dict a split, so a zero-valued "val" would still
    #    come back as an empty split (and log a warning about it); leaving it out
    #    means the dataset only ever has "train"/"test".
    datasets = {
        part: training.prepare_dataset(
            PROJECT_PATH, f"{DATASET_DIR}/{part}", part=part, from_part="organism",
            transforms=BODY_PART_TRANSFORMS, subset="body_parts_training",
            fractions={"train": 0.75, "test": 0.25},
        )
        for part in PARTS
    }

    # 5. Train one segmenter per part and register it. Registering is provenance
    #    only -- not an endorsement -- so every trained checkpoint gets registered
    #    here; step 6 below is what actually judges whether one is worth using.
    #
    #    val_split=None matches step 4's fractions (no "val" key): train() then
    #    trains every epoch with no early stopping and saves the final epoch's
    #    weights rather than the best-val-loss one -- the tradeoff for a reference
    #    set too small to spare a third split.
    for part in PARTS:
        checkpoint = training.train(datasets[part], f"{DATASET_DIR}/{part}",
                                    val_split=None)
        cf.register_model(
            PROJECT_PATH, f"{part}_segmenter_v1", path=checkpoint,
            task="segment", framework="torch",
            base_model=segmentation.DEFAULT_ENCODER,
            training_data=f"{DATASET_DIR}/{part}",
        )

    # 6. Validate each trained model against its held-out TEST split before
    #    deciding whether to wire it into the main pipeline.
    #
    #    validate_masks() compares masks already on disk -- CANONICAL against
    #    REFERENCE -- it doesn't run a model itself. So this first runs the
    #    freshly trained model over just the test occurrences (never seen during
    #    training or early stopping) and writes canonical part masks for them,
    #    then compares those against the reference masks made by hand in step 3.
    #    A distinct run_name from step 3's future main-pipeline run ("body_parts")
    #    is deliberate: these are a few throwaway test predictions for scoring a
    #    checkpoint, not meant to be mistaken for that run's own coverage.
    # for part in PARTS:
    #     test_ids = datasets[part].loc[datasets[part]["split"] == "test", "occurrence_id"]
    #     if test_ids.empty:
    #         logger.warning("no test-split occurrences for '%s' -- grow the "
    #                        "reference set or the fractions passed to "
    #                        "prepare_dataset()", part)
    #         continue
    #
    #     test_subset = f"{part}_test"
    #     cf.define_subset(PROJECT_PATH, test_subset, occurrence_ids=test_ids)
    #
    #     cf.run_segments(
    #         PROJECT_PATH,
    #         run_name=f"{part}_test_predictions",
    #         from_part="organism",
    #         part=part,
    #         subset=test_subset,
    #         shared_steps=BODY_PART_TRANSFORMS,
    #         steps=[cf.segment(load_part_segmenter(f"{part}_segmenter_v1"))],
    #         visualize=False,
    #     )
    #
    #     iou = cf.validate_masks(PROJECT_PATH, part=part, visualize=True)
    #     logger.info("%s: held-out test IoU -- mean %.3f over %d occurrence(s)",
    #                part, iou["iou"].mean() if len(iou) else float("nan"), len(iou))

    # Once the IoU each part reports above (uncomment step 6 to check) looks
    # good enough to trust, run scripts/dragonfly_bodies_inat.py -- its step 3
    # already imports BODY_PART_TRANSFORMS and load_part_segmenter from this
    # module.
