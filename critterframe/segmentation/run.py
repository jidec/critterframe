"""segment() and run_segments(), including sharded parallel runs."""

import logging
from collections import Counter

import numpy as np

from .. import drivers, segments as segment_iteration
from .. import selectionhelpers
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segmentation
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
    """Operation: derive a mask for the segment using a model.

    The model must provide `predict(image, mask_threshold=0.0) -> (mask, score, info)`:
    an RGB image in, a boolean mask of the same size, a confidence or None, and a
    diagnostics dict. Its `identity() -> dict`, if it has one, is what reaches the recipe
    hash; `visualize(...)` is optional.

    Args:
        model: A model meeting that contract.
        mask_threshold: Logit cutoff passed to the model: 0.0 is neutral, negative grows
            the mask, positive shrinks it.
    """
    return Segmentation("segment", _segment, {"mask_threshold": mask_threshold}, version="1", model=model)


def _segment(segment_state, model, mask_threshold=0.0):
    """Run the model over the segment's image and attach the mask it returns."""
    try:
        mask, score, info = model.predict(segment_state.rgb, mask_threshold=mask_threshold)
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
    """Emit the panel every segmentation contributes: the mask it settled on, over its frame."""
    if state.panel_sink is None or state.mask is None:
        return

    panel = overlay_mask(state.image, state.mask)
    area = int(state.mask.sum())
    annotate(panel, f"{state.part} {area}px ({area / max(1, state.mask.size):.1%})")
    if score is not None:
        annotate(panel, f"score {score:.3f}", line=1)
    state.emit_panel(panel, "mask")


def _upstream_parts(from_part):
    """Return the upstream part names `from_part` asks for, sorted and deduplicated."""
    if from_part is None:
        return []
    if isinstance(from_part, str):
        return [from_part]
    parts = sorted({str(part) for part in from_part})
    if not parts:
        raise ValueError("from_part names no part -- pass a part name, several, or None")
    return parts


def _upstream_mask(source_rows, occurrence_id):
    """Return the union of every upstream part's mask for one occurrence.

    Args:
        source_rows: `{upstream part: {occurrence_id: mask row}}`.
        occurrence_id: The occurrence.

    Raises:
        NoInput: If an upstream part has no mask for it.
    """
    rows = []
    for upstream, lookup in source_rows.items():
        row = lookup.get(occurrence_id)
        if row is None:
            raise drivers.NoInput(drivers.no_mask(upstream))
        rows.append((upstream, row))

    combined = None
    for upstream, row in rows:
        mask = mask_records.decode_mask(row)
        if combined is None:
            combined = mask
        elif mask.shape != combined.shape:
            # Padding would store a mask that no longer matches the image.
            raise ValueError(
                f"the '{upstream}' mask is {mask.shape} but another upstream "
                f"part's is {combined.shape} -- masks of one occurrence must "
                "share the original image's shape to be merged"
            )
        else:
            combined = combined | mask
    return combined


def _build_recipes(run_name, steps, outputs, shared_steps, part, from_part, reference, from_reference=False):
    """Turn `run_segments`' arguments into `{part: Recipe}`.

    A recipe's name defaults to its part, or `<part>_reference` for a reference pass.
    """
    if steps is not None and outputs is not None:
        raise ValueError("run_segments takes either steps= or outputs=, not both")
    if steps is None and outputs is None:
        raise ValueError("run_segments needs steps= (one part) or outputs= (several)")

    shared = list(shared_steps or [])
    inputs = {"masks": "reference" if reference else "canonical"}
    if from_reference:
        # Added only when set, so every recipe hash recorded before the flag
        # existed still holds.
        inputs["from_masks"] = "reference"

    def default_name(output_part):
        return f"{output_part}_reference" if reference else output_part

    if steps is not None:
        name = run_name if run_name is not None else default_name(part)
        return {
            part: Recipe("segment", name, shared + list(steps), part=part, from_part=from_part, inputs=inputs)
        }

    return {
        output_part: Recipe(
            "segment",
            run_name if run_name is not None else default_name(output_part),
            shared + list(output_steps),
            part=output_part,
            from_part=from_part,
            inputs=inputs,
        )
        for output_part, output_steps in outputs.items()
    }


def _failure_row(occurrence_id, part, recipe_hash, source_mask_hash, error):
    """Build one failures-ledger row, keyed by the recipe and the upstream mask."""
    return {
        "occurrence_id": occurrence_id,
        "part": part,
        "context_hash": mask_records.derivation_hash(recipe_hash, source_mask_hash),
        "error": str(error),
    }


def _resolve_force(force, recipes):
    """Return `force` as a bool.

    Raises:
        ValueError: If `force` is None and a recipe has a non-deterministic operation.
    """
    if force is not None:
        return bool(force)

    named = sorted(
        {operation.name for recipe in recipes.values() for operation in recipe.nondeterministic_operations()}
    )
    if not named:
        return False

    raise ValueError(
        f"{', '.join(named)} is not deterministic -- running it again can "
        "produce a different mask, so this run will not decide on its own "
        "whether the occurrence-parts it has already covered count as done. "
        "Pass force=False to keep what is recorded and continue where you "
        "stopped, or force=True to redo them."
    )


def run_segments(
    project_path,
    steps=None,
    run_name=None,
    part=DEFAULT_PART,
    outputs=None,
    shared_steps=None,
    from_part=None,
    subset=None,
    limit=None,
    force=None,
    visualize=True,
    visualize_every=None,
    reference=False,
    batch_size=DEFAULT_BATCH_SIZE,
    shard=None,
    retry_failed=False,
    from_reference=False,
):
    """Run a segmentation recipe over a project's occurrences.

    Three forms:

        run_segments(project_path, steps=[segment(model)])

        run_segments(project_path, from_part="organism",
                     shared_steps=[remove_background(), orient()],
                     outputs={"head": [segment(head_model)],
                              "abdomen": [segment(abdomen_model)]})

        run_segments(project_path, part="body",
                     from_part=["head", "thorax", "abdomen"], steps=[])

    Args:
        project_path: Project to process.
        steps: Ordered operations producing one part's mask. Use this or `outputs`.
        run_name: Name for the run; the part it produces if None, or `<part>_reference`
            with `reference`.
        part: Part that `steps` produces.
        outputs: `{part: steps}`, for several parts in one pass. Each part gets its own
            recipe and run.
        shared_steps: Operations run once per occurrence before the per-part steps.
        from_part: Start each segment from this part's mask. A list starts from the union
            of several; an occurrence missing any of them counts as `no_input`.
        subset: Named subset to process.
        limit: Cap on occurrences considered.
        force: Redo occurrence-parts this recipe already covered. None means False, except
            for a recipe with a non-deterministic operation, where it raises.
        visualize: True, an int, or ids: how much of a pipeline grid to write. False
            writes none. A fully cached rerun writes none either.
        visualize_every: Also write a checkpoint grid every N occurrences, sampled from
            that window.
        reference: Write to the reference mask table.
        batch_size: Masks accumulated before each write.
        shard: `(index, total)`: process only this shard of the occurrences. Writes are
            staged for `merge_mask_shards()`, failures are not recorded, and every shard's
            grid shares one filename, so pass `visualize=False`.
        retry_failed: Attempt occurrence-parts already recorded as failed under this recipe
            and upstream mask.
        from_reference: Read the `from_part` masks from the reference table.

    Returns:
        `{part: summary}`, each a `drivers.Tally.summary` plus `run_id`,
        `previously_failed` and `elapsed_s`.
    """
    paths.require_project(project_path)

    upstream_parts = _upstream_parts(from_part)
    if from_reference and not upstream_parts:
        raise ValueError("from_reference=True needs a from_part to read")
    # One upstream stays a plain name, so a recipe written with a single part
    # hashes as it always has.
    from_part = (
        None if not upstream_parts else (upstream_parts[0] if len(upstream_parts) == 1 else upstream_parts)
    )
    from_label = "+".join(upstream_parts) or None

    recipes = _build_recipes(
        run_name, steps, outputs, shared_steps, part, from_part, reference, from_reference
    )
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset, limit=limit)

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
                shard,
                visualize,
            )

    force = _resolve_force(force, recipes)

    # Reference and canonical passes record failures under their own stage:
    # records.failures keys on (occurrence_id, part, stage), so one stage for
    # both would have a reference failure upsert over the canonical one, and a
    # canonical success's clear_failures delete the reference's record.
    stage = "segment_reference" if reference else "segment"

    logger.info(
        "run_segments %s: %d occurrence(s), part(s): %s",
        {output_part: recipe.name for output_part, recipe in recipes.items()},
        len(occurrence_ids),
        ", ".join(sorted(recipes)),
    )

    # The upstream masks a from_part recipe starts from, loaded BEFORE the
    # pending check rather than alongside the images, because what still needs
    # doing depends on which upstream mask each part would be cut out of: a
    # derived part goes stale the moment the part it came from is resegmented,
    # while its own recipe hash sits there unchanged.
    source_rows = {
        upstream: mask_records.mask_lookup(
            project_path, part=upstream, occurrence_ids=occurrence_ids, reference=from_reference
        )
        for upstream in upstream_parts
    }
    # Only an occurrence holding every upstream part has a source to hash; one
    # missing any of them is never complete and is counted as no_input below.
    source_hashes = {}
    for occurrence_id in occurrence_ids if upstream_parts else ():
        rows = [lookup.get(occurrence_id) for lookup in source_rows.values()]
        if all(row is not None for row in rows):
            source_hashes[occurrence_id] = mask_records.combined_source_hash(
                {
                    upstream: mask_records.derivation_hash(row["recipe_hash"], row.get("source_mask_hash"))
                    for upstream, row in zip(source_rows, rows)
                }
            )

    # Which occurrence-parts still need work, per part. Computed up front so a
    # fully-cached run does no image loading at all rather than loading every
    # image and discarding it.
    pending = {}
    failed_by_part = {}
    for output_part, recipe in recipes.items():
        upstream = (
            None
            if from_part is None
            else {
                (occurrence_id, output_part): source_hash
                for occurrence_id, source_hash in source_hashes.items()
            }
        )
        done = (
            set()
            if force
            else mask_records.completed_keys(
                project_path, recipe.hash, reference=reference, source_mask_hashes=upstream
            )
        )

        failed = set()
        if not force and not retry_failed:
            context_hashes = {
                (occurrence_id, output_part): mask_records.derivation_hash(
                    recipe.hash, source_hashes.get(occurrence_id)
                )
                for occurrence_id in occurrence_ids
            }
            failed = failure_records.failed_keys(project_path, stage, context_hashes)
        failed_by_part[output_part] = failed

        skip = done | failed
        pending[output_part] = [
            occurrence_id for occurrence_id in occurrence_ids if (occurrence_id, output_part) not in skip
        ]
        logger.info(
            "  %s: %d pending, %d already done by recipe %s, "
            "%d previously failed (retry_failed=True to retry)",
            output_part,
            len(pending[output_part]),
            len(done),
            recipe.hash,
            len(failed),
        )

    # The covered set is this run row's own -- for a sharded run that is the
    # shard's slice, which is what this row actually processed.
    run_context = {
        "occurrences": ids_record(occurrence_ids),
        "limit": limit,
        "shard": None if shard is None else list(shard),
    }
    run_ids = {
        output_part: run_records.start_run(project_path, recipe, subset=subset, context=run_context)
        for output_part, recipe in recipes.items()
    }

    shared = list(shared_steps or [])
    # Every step's scalar info is stored with the mask it helped make, keyed by
    # its label; the shared steps' labels are the same in every part's recipe.
    labels = {
        output_part: segment_iteration.operation_labels(recipe.operations)
        for output_part, recipe in recipes.items()
    }
    shared_labels = next(iter(labels.values()))[: len(shared)]
    tallies = {}
    for output_part, ids in pending.items():
        tally = drivers.Tally(attempted=len(occurrence_ids))
        tally.skipped = len(occurrence_ids) - len(ids)
        tallies[output_part] = tally
    batches = {output_part: [] for output_part in recipes}
    failure_batches = {output_part: [] for output_part in recipes}
    resolved_keys = {output_part: [] for output_part in recipes}

    # Walked in occurrence-table order, and only for occurrences some part
    # still needs -- so one pass over the image store covers every part.
    pending_sets = {output_part: set(ids) for output_part, ids in pending.items()}
    todo = [
        occurrence_id
        for occurrence_id in occurrence_ids
        if any(occurrence_id in ids for ids in pending_sets.values())
    ]

    # One report per part, because each part has its own recipe. Every report
    # walks the whole todo list, so checkpoint windows line up across parts,
    # but only samples the occurrences its own part still needs: a grid can
    # only show work that happened (pass force=True to see a cached run again).
    reports = {
        output_part: pipeline_visualization.open_report(
            project_path,
            recipe.name,
            recipe.hash,
            part=output_part,
            visualize=visualize,
            visualize_every=visualize_every,
            identity=recipe.spec(),
        ).begin(todo, eligible=pending_sets[output_part])
        for output_part, recipe in recipes.items()
    }
    # The shared steps run once on a segment that then forks per part, so their
    # panels belong to every part's grid -- they're in every part's recipe.
    shared_sink = pipeline_visualization.PanelFanout(reports.values())

    def flush(output_part):
        rows = batches[output_part]
        if rows:
            if shard is not None:
                mask_records.save_mask_shard(project_path, rows, output_part, reference=reference)
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
                failure_records.record_failures(project_path, stage, failure_rows)
                failure_rows.clear()

            resolved = resolved_keys[output_part]
            if resolved:
                failure_records.clear_failures(project_path, stage, resolved)
                resolved.clear()

    missing = {output_part: Counter() for output_part in recipes}

    # One step per occurrence rather than per part: the shared steps and the
    # image load are paid once whatever the number of parts.
    progress = drivers.Progress(
        len(todo),
        f"run_segments part(s) {', '.join(sorted(recipes))}",
        tallies=tallies.values(),
        log=logger.info,
    )

    def item_done(occurrence_id):
        for report in reports.values():
            report.done(occurrence_id)
        progress.step()

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in todo:
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise drivers.NoInput(drivers.NO_IMAGE)

                start_mask = None
                if upstream_parts:
                    start_mask = _upstream_mask(source_rows, occurrence_id)

                base = segment_iteration.build_segment(
                    image,
                    mask=start_mask,
                    occurrence_id=occurrence_id,
                    part=from_label or DEFAULT_PART,
                    project_path=project_path,
                    panel_sink=pipeline_visualization.panel_sink(shared_sink, occurrence_id),
                )

                shared_info = {}
                for label, operation in zip(shared_labels, shared):
                    base, info = operation(base)
                    shared_info[label] = segment_iteration.scalar_info(info)
                    for tally in tallies.values():
                        tally.record_flags(info)

            except drivers.NoInput as exc:
                for output_part in recipes:
                    if occurrence_id in pending_sets[output_part]:
                        tallies[output_part].no_input += 1
                        missing[output_part][str(exc)] += 1
                item_done(occurrence_id)
                continue
            except Exception as exc:
                logger.warning("segmentation setup failed for %s: %s", occurrence_id, exc)
                for output_part in recipes:
                    if occurrence_id in pending_sets[output_part]:
                        tallies[output_part].record_failure(occurrence_id, exc)
                        reports[output_part].failure(occurrence_id, exc)
                        failure_batches[output_part].append(
                            _failure_row(
                                occurrence_id,
                                output_part,
                                recipes[output_part].hash,
                                source_hashes.get(occurrence_id),
                                exc,
                            )
                        )
                item_done(occurrence_id)
                continue

            for output_part, recipe in recipes.items():
                if occurrence_id not in pending_sets[output_part]:
                    continue
                try:
                    state = base.for_part(output_part)
                    # Past the fork, panels are this part's alone: the fanout
                    # was only right while the work was genuinely shared.
                    state.panel_sink = reports[output_part].sink(occurrence_id)
                    score = None
                    mask_info = dict(shared_info)
                    own = zip(labels[output_part][len(shared) :], recipe.operations[len(shared) :])
                    for label, operation in own:
                        state, info = operation(state)
                        mask_info[label] = segment_iteration.scalar_info(info)
                        tallies[output_part].record_flags(info)
                        if operation.kind == "segment" and info.get("score") is not None:
                            score = info["score"]

                    _visualize_result(state, score)

                    batches[output_part].append(
                        mask_records.make_mask_row(
                            occurrence_id,
                            state.mask_in_original_coordinates(),
                            part=output_part,
                            recipe_hash=recipe.hash,
                            run_id=run_ids[output_part],
                            score=score,
                            from_part=from_label,
                            source_mask_hash=source_hashes.get(occurrence_id),
                            info=mask_info,
                        )
                    )
                    tallies[output_part].processed += 1
                    resolved_keys[output_part].append((occurrence_id, output_part))

                    if len(batches[output_part]) >= batch_size:
                        flush(output_part)

                except Exception as exc:
                    tallies[output_part].record_failure(occurrence_id, exc)
                    reports[output_part].failure(occurrence_id, exc)
                    logger.warning(
                        "segmentation failed for %s part '%s': %s", occurrence_id, output_part, exc
                    )
                    failure_batches[output_part].append(
                        _failure_row(
                            occurrence_id, output_part, recipe.hash, source_hashes.get(occurrence_id), exc
                        )
                    )

            item_done(occurrence_id)

    for report in reports.values():
        report.close()
    elapsed = progress.finish()

    counts = {}
    for output_part in recipes:
        drivers.log_no_input(missing[output_part], f"run_segments part '{output_part}'")
        flush(output_part)
        tally = tallies[output_part]
        run_records.finish_run(
            project_path,
            run_ids[output_part],
            processed=tally.processed,
            skipped=tally.skipped,
            failed=tally.failed,
            flags=tally.flags,
        )
        counts[output_part] = tally.summary(
            run_id=run_ids[output_part], previously_failed=len(failed_by_part[output_part]), elapsed_s=elapsed
        )

    return counts
