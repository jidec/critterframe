"""
segment() operation + run_segments(), including sharded/parallel runs.

Two forms:

    # one part (the whole organism, by default)
    run_segments(project_path, steps=[segment(groundedsam2())])

    # several parts from one shared starting point
    run_segments(project_path, run_name="body_parts", from_part="organism",
                 shared_steps=[remove_background(), orient()],
                 outputs={"head": [segment(head_model)],
                          "abdomen": [segment(abdomen_model)]})

Shared steps run ONCE per occurrence and the segment forks per part, so a
three-part run does one background removal rather than three. Each part still
gets its own recipe and run record, so changing the abdomen model leaves head
and thorax masks alone.
"""

import logging

import numpy as np

from .. import selectionhelpers
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment, Segmentation
from ..records import failures as failure_records
from ..records import masks as mask_records
from ..records import runs as run_records
from ..records.occurrences import ids_record
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

    A model meets the segmenter contract if it exposes:

    ```
    predict(image) -> (mask, score, info)
    ```

    where image is RGB, mask is boolean of the same height/width, score is
    the model's confidence or None, and info is diagnostics. Optionally also
    `identity() -> dict` and `visualize(...)`.

    `identity()` is how a checkpoint reaches the recipe hash; without one a
    model is identified only by its class name, so two fine-tunes would be
    mistaken for equivalent work.

    - `model` -- anything meeting the segmenter contract above.
    - `mask_threshold` -- cutoff passed to models that take one. For SAM2 it
      is a LOGIT, not a probability: 0.0 is the neutral default, negative
      grows the mask, positive shrinks it. A tenth of a logit is a
      meaningful step.
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

    Drawn by the run rather than the model, because it is the one view that
    always exists -- a segmenter with no visualize() of its own would otherwise
    put nothing on the grid, which is exactly when you most want to look.
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

    The single- and multi-output forms differ only here; everything after
    handles a dict of recipes either way.

    run_name=None defaults each recipe's name to the part it produces --
    "organism", "head" -- rather than one generic label shared by every part,
    since name isn't part of identity (Recipe.hash) and exists only for a
    human reading history to have something better than a bare hash to go on.
    A reference pass gets "<part>_reference" instead of plain "<part>": with
    nothing said, a canonical and a reference recipe over the same part would
    otherwise default to the identical name, and since resolve_recipe_currency
    is a no-op for segments nothing would catch the collision -- history and
    describe_run(name=...) would silently read as one recipe superseding the
    other rather than two meant to coexist for comparison (the same reasoning
    CLAUDE.md already gives for why canonical and reference METRICS need
    their own names).
    """
    if steps is not None and outputs is not None:
        raise ValueError("run_segments takes either steps= or outputs=, not both")
    if steps is None and outputs is None:
        raise ValueError("run_segments needs steps= (one part) or outputs= (several)")

    shared = list(shared_steps or [])
    inputs = {"masks": "reference" if reference else "canonical"}

    def default_name(output_part):
        return f"{output_part}_reference" if reference else output_part

    if steps is not None:
        name = run_name if run_name is not None else default_name(part)
        return {part: Recipe("segment", name, shared + list(steps), part=part,
                             from_part=from_part, inputs=inputs)}

    return {
        output_part: Recipe("segment",
                            run_name if run_name is not None else default_name(output_part),
                            shared + list(output_steps), part=output_part,
                            from_part=from_part, inputs=inputs)
        for output_part, output_steps in outputs.items()
    }


def _resolve_force(force, recipes):
    """
    Settle whether this run may skip completed work, refusing to guess for a
    recipe that won't reproduce itself.

    force   -- run_segments' force argument, None where the caller said nothing.
    recipes -- {part: Recipe} about to run.

    Returns force as a bool.
    """
    if force is not None:
        return bool(force)

    named = sorted({operation.name for recipe in recipes.values()
                    for operation in recipe.nondeterministic_operations()})
    if not named:
        return False

    raise ValueError(
        f"{', '.join(named)} is not deterministic -- running it again can "
        "produce a different mask, so this run will not decide on its own "
        "whether the occurrence-parts it has already covered count as done. "
        "Pass force=False to keep what is recorded and continue where you "
        "stopped, or force=True to redo them."
    )


def run_segments(project_path, steps=None, run_name=None, part=DEFAULT_PART,
                 outputs=None, shared_steps=None, from_part=None, subset=None,
                 limit=None, force=None, visualize=True, visualize_every=None,
                 reference=False, batch_size=DEFAULT_BATCH_SIZE, shard=None,
                 retry_failed=False):
    """
    Run a segmentation recipe over a project's occurrences.

    - `project_path` -- the project to process.
    - `steps` -- ordered operations producing one part's mask. Use this OR
      `outputs`.
    - `run_name` -- what to call this run; recorded on the run, not part of
      recipe identity (see Recipe.hash) -- renaming costs nothing, since
      nothing looks a mask up by run_name. Defaults to `part` (or each
      output's own part, for `outputs=`), and to `<part>_reference` when
      `reference=True`, so the common case -- one segmentation recipe per
      part -- needs no name at all, and a canonical pass and a reference
      pass over the same part never collide on one default. Pass one
      explicitly to keep two coexisting recipes for one part apart in
      history, e.g. comparing two candidate models
      (`dragonfly_bodies_inat_training.py` does this by hand with
      `f"{part}_test_predictions"`).
    - `part` -- which part `steps` produces; the whole organism by default.
    - `outputs` -- `{part: steps}` for producing several parts in one pass.
    - `shared_steps` -- operations run once per occurrence before forking
      into each output's own steps.
    - `from_part` -- start each segment from an existing part's mask instead
      of from none. How refinement chains work: a part-specific model starts
      from the organism mask rather than rediscovering it. Which upstream
      mask each output came from is recorded, so resegmenting the upstream
      recomputes everything below it.
    - `subset` -- name of a subset to process, or None for every occurrence.
    - `limit` -- optional cap on occurrences, for trying a recipe out.
    - `force` -- redo occurrence-parts this recipe already covered from the
      same upstream mask. A part whose upstream has been replaced is redone
      regardless. None (the default) means False for a deterministic recipe;
      where an operation is `deterministic=False` it raises instead, since
      neither answer is safe to assume.
    - `visualize` -- how much of a pipeline grid to produce: 25 samples 25
      occurrences, True (default) uses the default sample size, a list names
      occurrences specifically, False produces none. A grid can only show
      work that happened, so a fully cached rerun writes none -- use
      `force=True` to see it again.
    - `visualize_every` -- in addition to the one grid above (sampled once
      from the whole run and comparable cell-by-cell across two recipe
      versions), write a second, independent series of checkpoint grids
      every N occurrences processed: one file per checkpoint, each resampled
      fresh from whatever that checkpoint's window actually processed, so it
      always has real content even when the whole-run sample above hasn't
      been reached yet. None (default) writes none of these; a killed run
      still leaves the checkpoint for its last completed window. No effect
      without `visualize`.
    - `reference` -- write to the reference mask table instead of the
      canonical one, as a human-drawn validation pass does.
    - `batch_size` -- masks accumulated before each write.
    - `shard` -- (index, total): process only this shard of the occurrences,
      for running several workers over one project at once. Shards are
      deterministic and disjoint, so workers need no coordination. A sharded
      run stages its writes; call `merge_mask_shards()` once afterwards.
      Note that every shard's QC grid shares one filename, so pass
      `visualize=False`. A sharded run does not persist failures (see
      `records.failures`) for the same reason it doesn't upsert
      `masks.parquet` directly -- failed occurrence-parts are still logged
      and counted, just retried on the next run.
    - `retry_failed` -- attempt occurrence-parts that already failed under
      this exact recipe (and, for a `from_part` recipe, this exact upstream
      mask). False (the default) leaves them recorded as failed; a changed
      recipe or upstream retries them automatically with no flag needed.

    Returns {part: {"processed", "skipped", "failed", "previously_failed", "run_id"}}.
    skipped includes occurrence-parts excluded for either reason -- already
    done, or already failed and not retried.
    """
    paths.require_project(project_path)

    recipes = _build_recipes(run_name, steps, outputs, shared_steps, part,
                             from_part, reference)
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                 limit=limit)

    if shard is not None:
        index, total = shard
        occurrence_ids = selectionhelpers.shard_occurrences(occurrence_ids, index, total)
        if visualize:
            logger.warning(
                "shard=%s with visualize=%r: every shard's QC grid(s) share the "
                "same filename(s) (including visualize_every's checkpoints), so "
                "only the last shard to write a given file will leave it on "
                "disk. Pass visualize=False for a sharded run to avoid the "
                "clobbering, or ignore this if only one shard's sample matters.",
                shard, visualize)

    if visualize_every and not visualize:
        logger.warning(
            "visualize_every=%d with visualize=%r: there's no report to "
            "checkpoint without visualize, so visualize_every has no effect",
            visualize_every, visualize)

    force = _resolve_force(force, recipes)

    logger.info("run_segments %s: %d occurrence(s), part(s): %s",
                {output_part: recipe.name for output_part, recipe in recipes.items()},
                len(occurrence_ids), ", ".join(sorted(recipes)))

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
    failed_by_part = {}
    for output_part, recipe in recipes.items():
        upstream = None if from_part is None else {
            (occurrence_id, output_part): source_hash
            for occurrence_id, source_hash in source_hashes.items()
        }
        done = set() if force else mask_records.completed_keys(
            project_path, recipe.hash, reference=reference,
            source_mask_hashes=upstream)

        failed = set()
        if not force and not retry_failed:
            context_hashes = {
                (occurrence_id, output_part): mask_records.derivation_hash(
                    recipe.hash, source_hashes.get(occurrence_id))
                for occurrence_id in occurrence_ids
            }
            failed = failure_records.failed_keys(project_path, "segment", context_hashes)
        failed_by_part[output_part] = failed

        skip = done | failed
        pending[output_part] = [
            occurrence_id for occurrence_id in occurrence_ids
            if (occurrence_id, output_part) not in skip
        ]
        logger.info("  %s: %d pending, %d already done by recipe %s, "
                    "%d previously failed (retry_failed=True to retry)",
                    output_part, len(pending[output_part]), len(done), recipe.hash,
                    len(failed))

    # The covered set is this run row's own -- for a sharded run that is the
    # shard's slice, which is what this row actually processed.
    run_context = {"occurrences": ids_record(occurrence_ids), "limit": limit,
                   "shard": None if shard is None else list(shard)}
    run_ids = {
        output_part: run_records.start_run(project_path, recipe, subset=subset,
                                           context=run_context)
        for output_part, recipe in recipes.items()
    }

    shared = list(shared_steps or [])
    counts = {
        output_part: {"processed": 0, "skipped": len(occurrence_ids) - len(ids),
                      "failed": 0, "previously_failed": len(failed_by_part[output_part])}
        for output_part, ids in pending.items()
    }
    batches = {output_part: [] for output_part in recipes}
    failure_batches = {output_part: [] for output_part in recipes}
    resolved_keys = {output_part: [] for output_part in recipes}

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
            project_path, recipe.name, recipe.hash, output_part,
            pending[output_part], visualize)
        for output_part, recipe in recipes.items()
    }
    # The shared steps run once on a segment that then forks per part, so their
    # panels belong to every part's grid -- they're in every part's recipe.
    active = [report for report in reports.values() if report is not None]
    shared_sink = pipeline_visualization.PanelFanout(active) if active else None

    # visualize_every's checkpoint series: a second RunReport per part,
    # resampled fresh each window from whatever that window actually
    # processes rather than the fixed whole-population sample above, so a
    # checkpoint always has real content -- see pipeline_visualization's
    # module docstring. window_reports/part_sinks/shared_sink are rebuilt by
    # rotate_windows() at every checkpoint; when visualize_every is unset this
    # is a no-op and part_sinks/shared_sink stay exactly what they are above.
    window_reports = {output_part: None for output_part in recipes}
    part_sinks = dict(reports)

    def rotate_windows(start_index):
        nonlocal shared_sink
        if not (visualize_every and visualize) or start_index >= len(todo):
            return
        end = min(start_index + visualize_every, len(todo))
        window_ids = todo[start_index:end]
        window_active = []
        for output_part in recipes:
            ids = [oid for oid in window_ids if oid in pending_sets[output_part]]
            sample = pipeline_visualization.resolve_sample(ids, visualize)
            report = None
            if sample:
                report = pipeline_visualization.RunReport(
                    project_path, recipes[output_part].name,
                    recipes[output_part].hash, output_part, sample,
                    suffix=f"__at{end:08d}")
                window_active.append(report)
            window_reports[output_part] = report
            combined = [existing for existing in (reports[output_part], report)
                       if existing is not None]
            part_sinks[output_part] = (
                pipeline_visualization.PanelFanout(combined) if len(combined) > 1
                else (combined[0] if combined else None))
        shared_sink = (pipeline_visualization.PanelFanout(active + window_active)
                      if (active or window_active) else None)

    rotate_windows(0)

    def flush(output_part):
        rows = batches[output_part]
        if rows:
            if shard is not None:
                mask_records.save_mask_shard(project_path, rows, output_part,
                                             reference=reference)
            else:
                mask_records.save_masks(project_path, rows, reference=reference)
            rows.clear()

        # A sharded run stages mask writes rather than upserting directly,
        # because upsert_table has no locking and concurrent writers would
        # silently lose each other's rows -- the same hazard applies to
        # failures.parquet, so a sharded run simply doesn't persist failures:
        # they're still logged and counted, just retried on the next run.
        if shard is None:
            failure_rows = failure_batches[output_part]
            if failure_rows:
                failure_records.record_failures(project_path, "segment", failure_rows)
                failure_rows.clear()

            resolved = resolved_keys[output_part]
            if resolved:
                failure_records.clear_failures(project_path, "segment", resolved)
                resolved.clear()

    def maybe_checkpoint(index):
        # Cheap even when nothing new was collected -- RunReport.save() is a
        # no-op then (see its dirty flag) -- so this is safe to call on every
        # occurrence's periodic boundary regardless of whether that occurrence
        # was itself in the sample.
        if visualize_every and (index + 1) % visualize_every == 0:
            for report in active:
                report.save()
            for report in window_reports.values():
                if report is not None:
                    report.save()
            rotate_windows(index + 1)

    with ImageStore(project_path, readonly=True) as images:
        for index, occurrence_id in enumerate(todo):
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
                        failure_batches[output_part].append({
                            "occurrence_id": occurrence_id,
                            "part": output_part,
                            "context_hash": mask_records.derivation_hash(
                                recipes[output_part].hash,
                                source_hashes.get(occurrence_id)),
                            "error": str(exc),
                        })
                maybe_checkpoint(index)
                continue

            for output_part, recipe in recipes.items():
                if occurrence_id not in pending_sets[output_part]:
                    continue
                try:
                    state = base.for_part(output_part)
                    # Past the fork, panels are this part's alone: the fanout
                    # was only right while the work was genuinely shared.
                    state.panel_sink = pipeline_visualization.panel_sink(
                        part_sinks[output_part], occurrence_id)
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
                    resolved_keys[output_part].append((occurrence_id, output_part))

                    if len(batches[output_part]) >= batch_size:
                        flush(output_part)

                except Exception as exc:
                    counts[output_part]["failed"] += 1
                    logger.warning("segmentation failed for %s part '%s': %s",
                                   occurrence_id, output_part, exc)
                    failure_batches[output_part].append({
                        "occurrence_id": occurrence_id,
                        "part": output_part,
                        "context_hash": mask_records.derivation_hash(
                            recipe.hash, source_hashes.get(occurrence_id)),
                        "error": str(exc),
                    })

            maybe_checkpoint(index)

    for report in active:
        report.save()
    # A run that completes without landing exactly on a checkpoint boundary
    # still leaves a grid for its trailing partial window.
    for report in window_reports.values():
        if report is not None:
            report.save()

    for output_part in recipes:
        flush(output_part)
        run_records.finish_run(project_path, run_ids[output_part],
                               processed=counts[output_part]["processed"],
                               skipped=counts[output_part]["skipped"],
                               failed=counts[output_part]["failed"])
        counts[output_part]["run_id"] = run_ids[output_part]

    return counts
