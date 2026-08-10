"""
Compose and run segmentation chains.

run_segments() is one of the two functions a pipeline script actually calls. It
takes an ordered list of operations, turns it into a recipe, and executes that
recipe over a project's occurrences, writing one canonical mask per
occurrence-part.

Two forms:

    # one part (the whole organism, by default)
    run_segments(project_path, steps=[segment(groundedsam2())])

    # several parts from one shared starting point
    run_segments(
        project_path,
        run_name="body_parts",
        from_part="organism",
        shared_steps=[remove_background(), orient()],
        outputs={
            "head":    [segment(head_model)],
            "thorax":  [segment(thorax_model)],
            "abdomen": [segment(abdomen_model)],
        },
    )

The multi-output form exists because parts of one organism usually share most
of their preprocessing and differ only at the end. Shared steps run ONCE per
occurrence and the resulting segment forks per part, so a three-part run does
one background removal and one orientation, not three.

Each output part still gets its OWN recipe and its own run record, hashed over
the shared steps plus that part's steps. That's what keeps repeat-awareness
per-part: if the abdomen model changes, abdomen masks are recomputed and head
and thorax masks are left alone.

Repeat-awareness follows derivation as well as configuration. A part cut out of
another part's mask records which mask that was, so resegmenting the organism
recomputes every part built on it -- an unchanged recipe is only half of "this
work is still good".
"""

import logging

import numpy as np

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment, Segmentation
from ..records import masks as mask_records
from ..records import runs as run_records
from ..storage.imagestore import ImageStore
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)

# How many masks to accumulate before writing them out. Writing rewrites the
# mask parquet, so flushing constantly is wasteful and never flushing loses a
# long run to one interruption; a few hundred is the compromise, and it's what
# makes a killed run resume from roughly where it stopped rather than from the
# start.
DEFAULT_BATCH_SIZE = 250


def segment(model, mask_threshold=0.0):
    """
    Operation: derive a mask for the current segment using a model.

    model -- anything meeting the segmenter contract:

               predict(image) -> (mask, score, info)

             where image is an RGB array, mask is a boolean array of the same
             height/width, score is the model's own confidence (or None), and
             info is a dict of diagnostics. Optionally also:

               identity() -> dict
               visualize(segment, image, mask, score, info) -> None

             identity() is how a model's checkpoint reaches the recipe hash; a
             model without one is identified only by its class name, so two
             different fine-tunes of the same class would be mistaken for
             equivalent work. Give a trained model an identity().

             segmentation.groundedsam.GroundedSAM2 meets all of this. So does a
             part-specific model you train yourself -- which is the intended
             path for head/thorax/abdomen segmenters: wrap your network in a
             small class with predict() and identity(), and it composes with
             everything here.

    mask_threshold -- cutoff passed to models that take one, in whatever units
                      that model thresholds in. For SAM2 (and so for
                      GroundedSAM2, the bundled one) it is a LOGIT, not a
                      probability: 0.0 is the model's own default and the
                      neutral choice, negative values grow the mask, positive
                      values shrink it. A tenth of a logit is a meaningful step;
                      0.5 is already a noticeably tighter mask, not the
                      "middle" that a probability-shaped 0.5 suggests.
    """
    return Segmentation("segment", _segment, {"mask_threshold": mask_threshold},
                        version="1", model=model)


def _segment(segment_state, model, mask_threshold=0.0):
    """Run a model over the segment's current image and attach the mask it returns."""
    try:
        mask, score, info = model.predict(segment_state.rgb,
                                          mask_threshold=mask_threshold)
    except TypeError:
        # A model that doesn't take a threshold is fine; not every segmenter
        # produces something thresholdable in the first place.
        mask, score, info = model.predict(segment_state.rgb)

    mask = np.asarray(mask) > 0
    if mask.shape != segment_state.shape:
        raise ValueError(
            f"model returned a {mask.shape} mask for a {segment_state.shape} "
            "frame -- a segmenter must return a mask matching the image it was "
            "given, so it stays alignable with the original coordinates"
        )
    if not mask.any():
        raise ValueError("model returned an empty mask")

    info = dict(info or {})
    info["score"] = None if score is None else float(score)
    info["area"] = int(mask.sum())
    info["area_fraction"] = float(mask.sum() / mask.size)

    result = segment_state.replace(mask=mask)
    if segment_state.panel_sink is not None and hasattr(model, "visualize"):
        model.visualize(segment_state, segment_state.rgb, mask, score, info)

    return result, info


def _visualize_result(state, score):
    """
    The panel every segmentation run contributes itself: the mask it settled on,
    over the frame it was found in.

    Drawn by the run rather than left to the model, because it's the one view
    that always exists and is always the point -- a segmenter with no visualize()
    hook of its own would otherwise put nothing on the grid, which is exactly
    when you most want to look.
    """
    if state.panel_sink is None or state.mask is None:
        return

    panel = overlay_mask(state.image, state.mask)
    area = int(state.mask.sum())
    annotate(panel, f"{state.part} {area}px "
                    f"({area / max(1, state.mask.size):.1%})")
    if score is not None:
        annotate(panel, f"score {score:.3f}", line=1)
    state.emit_panel(panel, "mask")


def _build_recipes(run_name, steps, outputs, shared_steps, part, from_part,
                   reference):
    """
    Turn the caller's arguments into {part: Recipe}.

    The single-output and multi-output forms differ only here; everything after
    this point handles a dict of recipes either way, so there's one execution
    path rather than two that have to be kept in step with each other.
    """
    if steps is not None and outputs is not None:
        raise ValueError("run_segments takes either steps= or outputs=, not both")
    if steps is None and outputs is None:
        raise ValueError("run_segments needs steps= (one part) or outputs= (several)")

    shared = list(shared_steps or [])
    inputs = {"masks": "reference" if reference else "canonical"}

    if steps is not None:
        return {part: Recipe("segment", run_name, shared + list(steps), part=part,
                             from_part=from_part, inputs=inputs)}

    return {
        output_part: Recipe("segment", run_name, shared + list(output_steps),
                            part=output_part, from_part=from_part, inputs=inputs)
        for output_part, output_steps in outputs.items()
    }


def run_segments(project_path, steps=None, run_name="segments", part=DEFAULT_PART,
                 outputs=None, shared_steps=None, from_part=None, subset=None,
                 limit=None, force=False, visualize=True, reference=False,
                 batch_size=DEFAULT_BATCH_SIZE):
    """
    Run a segmentation recipe over a project's occurrences.

    project_path -- the project to process.
    steps        -- ordered operations producing one part's mask. Use this OR
                    outputs.
    run_name     -- what to call this run: recorded on the run and part of
                    recipe identity. Named run_name rather than name because
                    several other names are in play at a call site -- a part's,
                    a subset's, a model's.
    part         -- which part `steps` produces; the whole organism by default.
    outputs      -- {part: steps} for producing several parts in one pass.
    shared_steps -- operations run once per occurrence before forking into each
                    output's own steps.
    from_part    -- start each segment from an existing part's mask instead of
                    from no mask. This is how refinement chains work: a
                    part-specific model starts from the organism mask rather
                    than rediscovering the organism. Occurrences with no mask
                    for that part are skipped with a warning -- there's nothing
                    to start from.

                    Which upstream mask each output was derived from is recorded
                    on it, so resegmenting the upstream part is enough to make
                    every part below it recompute on the next run: a chain stays
                    consistent without anyone having to remember what depends on
                    what.
    subset       -- name of a subset to process, or None for every occurrence.
    limit        -- optional cap on occurrences, for trying a recipe out.
    force        -- redo occurrence-parts this exact recipe already covered from
                    the same upstream mask. Normally those are skipped, which is
                    what makes an interrupted run resumable and a repeated run a
                    no-op; use this when you want the work redone anyway. A part
                    whose upstream mask has been replaced since is redone
                    regardless -- what's stored for it was cut out of a mask the
                    project no longer has.
    visualize    -- how much of a pipeline grid to produce (see
                    visualization.pipeline).

                      25          sample 25 of the occurrences this run
                                  processes and write ONE grid per part to
                                  visualizations/pipeline/, rows of specimens
                                  against columns of stages.
                      True        the same, with the default sample size.
                      ["a", "b"]  those occurrences specifically, for following
                                  a known-difficult specimen through a recipe.
                      False       none (default).

                    A grid can only show work that happened, so a rerun that
                    skips everything as already done produces no grid; use
                    force=True to see a cached recipe's output again.
    reference    -- write to the reference mask table instead of the canonical
                    one. What a human-drawn validation pass uses (see
                    segmentation.manual), so reference masks coexist with the
                    canonical ones rather than replacing them.
    batch_size   -- masks accumulated before each write.

    Returns {part: {"processed", "skipped", "failed", "run_id"}}.
    """
    paths.require_project(project_path)

    recipes = _build_recipes(run_name, steps, outputs, shared_steps, part,
                             from_part, reference)
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                 limit=limit)
    logger.info("run_segments '%s': %d occurrence(s), part(s): %s",
                run_name, len(occurrence_ids), ", ".join(sorted(recipes)))

    # The upstream masks a from_part recipe starts from, loaded BEFORE the
    # pending check rather than alongside the images, because what still needs
    # doing depends on which upstream mask each part would be cut out of: a
    # derived part goes stale the moment the part it came from is resegmented,
    # while its own recipe hash sits there unchanged.
    source_masks = {}
    if from_part is not None:
        source_masks = mask_records.mask_lookup(project_path, part=from_part,
                                                occurrence_ids=occurrence_ids)
    source_hashes = {
        occurrence_id: mask_records.derivation_hash(
            row["recipe_hash"], row.get("source_mask_hash"))
        for occurrence_id, row in source_masks.items()
    }

    # Which occurrence-parts still need work, per part. Computed up front so a
    # fully-cached run does no image loading at all rather than loading every
    # image and discarding it.
    pending = {}
    for output_part, recipe in recipes.items():
        upstream = None if from_part is None else {
            (occurrence_id, output_part): source_hash
            for occurrence_id, source_hash in source_hashes.items()
        }
        done = set() if force else mask_records.completed_keys(
            project_path, recipe.hash, reference=reference,
            source_mask_hashes=upstream)
        pending[output_part] = [
            occurrence_id for occurrence_id in occurrence_ids
            if (occurrence_id, output_part) not in done
        ]
        logger.info("  %s: %d pending, %d already done by recipe %s",
                    output_part, len(pending[output_part]),
                    len(occurrence_ids) - len(pending[output_part]), recipe.hash)

    run_ids = {
        output_part: run_records.start_run(project_path, recipe, subset=subset)
        for output_part, recipe in recipes.items()
    }

    shared = list(shared_steps or [])
    counts = {output_part: {"processed": 0, "skipped": len(occurrence_ids) - len(ids),
                            "failed": 0}
              for output_part, ids in pending.items()}
    batches = {output_part: [] for output_part in recipes}

    # Walked in occurrence-table order, and only for occurrences some part
    # still needs -- so one pass over the image store covers every part.
    pending_sets = {output_part: set(ids) for output_part, ids in pending.items()}
    todo = [
        occurrence_id for occurrence_id in occurrence_ids
        if any(occurrence_id in ids for ids in pending_sets.values())
    ]

    # One report per part, because one report is one grid and each part has its
    # own recipe. Sampled from what this run will actually process, not from
    # everything it was pointed at: a grid can only show work that happened, and
    # a fully cached rerun has none to show (pass force=True to see it again).
    reports = {
        output_part: pipeline_visualization.run_report(
            project_path, run_name, recipe.hash, output_part,
            pending[output_part], visualize)
        for output_part, recipe in recipes.items()
    }
    # The shared steps run once on a segment that then forks per part, so their
    # panels belong to every part's grid -- they're in every part's recipe.
    active = [report for report in reports.values() if report is not None]
    shared_sink = pipeline_visualization.PanelFanout(active) if active else None

    def flush(output_part):
        rows = batches[output_part]
        if rows:
            mask_records.save_masks(project_path, rows, reference=reference)
            rows.clear()

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in todo:
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                start_mask = None
                if from_part is not None:
                    source = source_masks.get(occurrence_id)
                    if source is None:
                        raise ValueError(
                            f"no '{from_part}' mask to start from -- segment "
                            f"that part first"
                        )
                    start_mask = mask_records.decode_mask(source)

                base = Segment(image, mask=start_mask, occurrence_id=occurrence_id,
                               part=from_part or DEFAULT_PART,
                               project_path=project_path,
                               panel_sink=pipeline_visualization.panel_sink(
                                   shared_sink, occurrence_id))

                for operation in shared:
                    base, _info = operation(base)

            except Exception as exc:
                logger.warning("segmentation setup failed for %s: %s",
                               occurrence_id, exc)
                for output_part in recipes:
                    if occurrence_id in pending_sets[output_part]:
                        counts[output_part]["failed"] += 1
                continue

            for output_part, recipe in recipes.items():
                if occurrence_id not in pending_sets[output_part]:
                    continue
                try:
                    state = base.for_part(output_part)
                    # Past the fork, panels are this part's alone: the fanout
                    # was only right while the work was genuinely shared.
                    state.panel_sink = pipeline_visualization.panel_sink(
                        reports[output_part], occurrence_id)
                    score = None
                    for operation in recipe.operations[len(shared):]:
                        state, info = operation(state)
                        if operation.kind == "segment" and info.get("score") is not None:
                            score = info["score"]

                    _visualize_result(state, score)

                    batches[output_part].append(mask_records.make_mask_row(
                        occurrence_id,
                        state.mask_in_original_coordinates(),
                        part=output_part,
                        recipe_hash=recipe.hash,
                        run_id=run_ids[output_part],
                        score=score,
                        from_part=from_part,
                        source_mask_hash=source_hashes.get(occurrence_id),
                    ))
                    counts[output_part]["processed"] += 1

                    if len(batches[output_part]) >= batch_size:
                        flush(output_part)

                except Exception as exc:
                    counts[output_part]["failed"] += 1
                    logger.warning("segmentation failed for %s part '%s': %s",
                                   occurrence_id, output_part, exc)

    for report in active:
        report.save()

    for output_part in recipes:
        flush(output_part)
        run_records.finish_run(project_path, run_ids[output_part],
                               processed=counts[output_part]["processed"],
                               skipped=counts[output_part]["skipped"],
                               failed=counts[output_part]["failed"])
        counts[output_part]["run_id"] = run_ids[output_part]

    return counts
