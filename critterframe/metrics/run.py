"""
Compose and run metric pipelines.

run_metrics() is the other function a pipeline script actually calls. It takes
an ordered list of transforms and a list of metrics, turns them into a recipe,
and executes that recipe over a project's occurrence-parts, storing one value
per metric per occurrence-part.

    run_metrics(
        project_path,
        run_name="traits",
        transforms=[remove_appendages(), orient()],
        metrics=[body_length(), max_width()],
    )

Transforms come first and metrics last because that IS the pipeline: transforms
compose a working segment, metrics read terminal values off the end of it.
Nothing composes onto a metric's output, which is why a metric can't appear in
the middle of a chain.

Transforms here are NOT persisted. A metric run that removes appendages and
orients is measuring a temporary working state, not rewriting the stored mask
-- appendage removal is right for measuring body length and wrong for measuring
total area, and both should be derivable from the same stored mask. What gets
persisted is the recipe that says how the measurement was taken, so the working
state is always reconstructible.
"""

import logging

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment
from ..records import masks as mask_records
from ..records import metrics as metric_records
from ..records import runs as run_records
from ..storage.imagestore import ImageStore
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)


class RunContext:
    """
    What a run tells its operations before the per-occurrence loop starts.

    Passed to every operation's prepare() hook. Almost all of them ignore it --
    a metric is normally self-contained per occurrence -- but group metrics need
    to know which project and which occurrences make up the reference population
    they're about to fit against, and this is how they find out (see
    metrics.outliers).

    project_path   -- project being processed.
    occurrence_ids -- ids this run covers, AFTER subset and limit are applied
                      but BEFORE skipping already-completed work. A group metric
                      must fit against the whole population it's scoring within,
                      not against whichever occurrences happen to be left over
                      from an interrupted earlier run -- otherwise resuming a
                      run would silently change what "outlier" means partway
                      through it.
    part           -- part being measured.
    run_name       -- name of the run.
    """

    def __init__(self, project_path, occurrence_ids, part, run_name):
        self.project_path = project_path
        self.occurrence_ids = list(occurrence_ids)
        self.part = part
        self.run_name = run_name


# Distinguishes "this occurrence-part has no mask" from "its mask has no recipe
# hash" when comparing against the source-mask map below. Both look like None
# otherwise, and treating the first as a match would let a value survive the
# disappearance of the thing it was measured from.
_NO_MASK = object()


def _format_value(value):
    """One metric value as short text for a panel caption."""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(value)[:3]) + "}"
    text = str(value)
    return text if len(text) <= 14 else text[:13] + "…"


def _visualize_measurement(state, rows):
    """
    The panel every metric run contributes itself: what was measured, on the
    segment it was measured from, with the numbers written on it.

    This is the view that makes a QC grid worth opening -- a number that looks
    wrong in a table is ambiguous, while a number written across the mask it
    came from usually explains itself immediately.
    """
    if state.panel_sink is None:
        return

    panel = overlay_mask(state.image, state.mask) if state.mask is not None \
        else state.image.copy()
    for line, row in enumerate(rows[:6]):
        annotate(panel, f"{row['metric_name']}={_format_value(row['value'])}",
                 line=line)
    state.emit_panel(panel, "measured")


def completed_keys(project_path, recipe_hash, source_mask_hashes=None):
    """
    The (occurrence_id, part) pairs a recipe has already produced values for --
    the repeat-awareness check this module makes before doing any work.

    It lives here rather than in records.metrics because it isn't a way of
    reading the metric log, it's this run's decision about what work is left:
    the records layer stores values and answers questions about them, and what
    counts as already done is a property of how a run is executed. (The query is
    a one-liner against the same table records.runs owns the schema for.)

    Deliberately keyed on the recipe hash and not the run: work already
    completed by an earlier, interrupted run of the SAME recipe is exactly the
    work a new run should skip, and it shouldn't matter that a different run_id
    did it.

    source_mask_hashes -- {(occurrence_id, part): derivation_hash} of the masks
                          the caller is about to measure, from
                          records.masks.current_derivation_hashes(). When given,
                          completion means "this recipe has run over THIS mask",
                          not merely "this recipe has run here": resegmenting an
                          occurrence makes every value derived from its old mask
                          stale, and a metric rerun has to recompute it rather
                          than skip the occurrence because a row with the right
                          recipe hash happens to exist. Rows the caller has no
                          mask for never count as complete -- the mask that
                          produced them is not one this project still holds.

                          Omit it to ask the weaker question ("has this recipe
                          produced anything here at all"), which is all a caller
                          with no masks in hand can ask.
    """
    with run_records.open_database(project_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT occurrence_id, part, source_mask_hash FROM metrics
            WHERE recipe_hash = ?
            """,
            (recipe_hash,),
        ).fetchall()

    if source_mask_hashes is None:
        return {(row["occurrence_id"], row["part"]) for row in rows}

    return {
        (row["occurrence_id"], row["part"])
        for row in rows
        if row["source_mask_hash"] == source_mask_hashes.get(
            (row["occurrence_id"], row["part"]), _NO_MASK)
    }


def run_metrics(project_path, run_name, metrics, transforms=(), part=DEFAULT_PART,
                parts=None, subset=None, limit=None, force=False,
                visualize=True, reference=False):
    """
    Run a metric recipe over a project's occurrence-parts.

    project_path -- the project to process.
    run_name     -- what to call this run. Recorded on the run, part of recipe
                    identity, and the first component of every exported column
                    name -- so two differently-configured measurements of the
                    same trait stay distinguishable in the export instead of
                    overwriting each other. Named run_name rather than name
                    because several other names are in play at a call site: a
                    metric's own name, a part's, a subset's.
    metrics      -- list of Metric operations to evaluate.
    transforms   -- ordered Transform operations applied before measuring.
    part         -- part to measure; the whole organism by default.
    parts        -- several parts to measure, each with the same recipe. Each
                    gets its own run record and its own recipe hash (part is
                    part of identity), so a re-run for one part doesn't touch
                    the others.
    subset       -- name of a subset to process, or None for every occurrence.
    limit        -- optional cap on occurrences.
    force        -- recompute occurrence-parts this exact recipe already covered
                    FROM THE MASK IT IS ABOUT TO MEASURE. Normally those are
                    skipped, which is what makes an expensive metric behave like
                    cached derived data rather than a calculation redone on
                    every invocation. Occurrences whose mask has been replaced
                    since their value was computed are recomputed anyway, force
                    or not: their stored value describes a mask that no longer
                    exists, so there is no cached work there to reuse.
    visualize    -- 25 for a sampled QC grid in visualizations/pipeline/, True
                    for one of default size, a list of occurrence ids for those
                    specimens specifically, False for none. It means exactly
                    what it means in run_segments -- every operation may
                    contribute a column -- and the last column of a metric grid
                    is the measured segment with its values written on it.
    reference    -- measure the reference masks instead of the canonical ones.
                    This is how reference METRIC values are produced for
                    validation: the same recipe, pointed at the reference mask
                    table, so any disagreement is attributable to the masks
                    rather than to a different measurement method.

    Returns {part: {"processed", "skipped", "failed", "run_id"}}.
    """
    paths.require_project(project_path)

    metrics = list(metrics)
    transforms = list(transforms)
    if not metrics:
        raise ValueError("run_metrics needs at least one metric")
    for operation in metrics:
        if operation.kind != "metric":
            raise TypeError(
                f"metrics= got a {operation.kind} operation ({operation.name}) "
                "-- transforms belong in transforms="
            )
    for operation in transforms:
        if operation.kind == "metric":
            raise TypeError(
                f"transforms= got a metric operation ({operation.name}) -- "
                "metrics produce terminal values and can't be composed onto"
            )

    target_parts = list(parts) if parts else [part]
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                 limit=limit)
    logger.info("run_metrics '%s': %d occurrence(s), part(s): %s",
                run_name, len(occurrence_ids), ", ".join(target_parts))

    results = {}
    for target_part in target_parts:
        results[target_part] = _run_one_part(
            project_path, run_name, metrics, transforms, target_part,
            occurrence_ids, subset, force, visualize, reference,
        )
    return results


def _run_one_part(project_path, run_name, metrics, transforms, part,
                  occurrence_ids, subset, force, visualize, reference):
    """Execute one part's recipe. Split out so the multi-part loop stays readable."""
    recipe = Recipe("metric", run_name, transforms + metrics, part=part,
                    inputs={"masks": "reference" if reference else "canonical"})

    recipe.prepare_all(RunContext(project_path, occurrence_ids, part, run_name))

    mask_rows = mask_records.mask_lookup(project_path, part=part,
                                         occurrence_ids=occurrence_ids,
                                         reference=reference)

    # Completion is per (recipe, mask), not per recipe: a value already computed
    # by this recipe from a mask that has since been resegmented describes a
    # mask this run isn't measuring, so it can't stand in for the work. The
    # hashes come from the rows already loaded above, so this costs no extra
    # read. derivation_hash rather than the bare recipe hash so that a part
    # derived from another part is seen to change when its upstream does, which
    # the wing's own unchanged recipe hash would otherwise hide.
    source_hashes = {
        (occurrence_id, part): mask_records.derivation_hash(
            row["recipe_hash"], row.get("source_mask_hash"))
        for occurrence_id, row in mask_rows.items()
    }
    done = set() if force else completed_keys(project_path, recipe.hash,
                                              source_mask_hashes=source_hashes)

    todo = [
        occurrence_id for occurrence_id in occurrence_ids
        if occurrence_id in mask_rows and (occurrence_id, part) not in done
    ]
    unsegmented = len([i for i in occurrence_ids if i not in mask_rows])
    skipped = len(occurrence_ids) - len(todo) - unsegmented

    logger.info("  %s: %d to measure, %d already done by recipe %s over the "
                "same mask, %d with no mask", part, len(todo), skipped,
                recipe.hash, unsegmented)
    if unsegmented:
        logger.warning("  %s: %d occurrence(s) have no '%s' mask -- run "
                       "segmentation for that part first", part, unsegmented, part)

    report = pipeline_visualization.run_report(project_path, run_name,
                                               recipe.hash, part, todo,
                                               visualize)

    run_id = run_records.start_run(project_path, recipe, subset=subset)
    processed = 0
    failed = 0

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in todo:
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                mask_row = mask_rows[occurrence_id]
                state = Segment(image,
                                mask=mask_records.decode_mask(mask_row),
                                occurrence_id=occurrence_id, part=part,
                                project_path=project_path,
                                panel_sink=pipeline_visualization.panel_sink(
                                    report, occurrence_id))

                for operation in transforms:
                    state, _info = operation(state)

                rows = [
                    metric_records.make_metric_row(
                        occurrence_id, part, operation.metric_name,
                        operation(state), unit=operation.unit,
                        source_mask_hash=source_hashes[(occurrence_id, part)],
                    )
                    for operation in metrics
                ]

                _visualize_measurement(state, rows)

                # Written per occurrence rather than buffered to the end: a
                # human-annotation run can be hours of clicking, and an
                # interruption should cost the current occurrence, not the day.
                metric_records.append_metrics(project_path, run_id, recipe.hash, rows)
                processed += 1

            except Exception as exc:
                failed += 1
                logger.warning("metrics failed for %s part '%s': %s",
                               occurrence_id, part, exc)

    if report is not None:
        report.save()

    run_records.finish_run(project_path, run_id, processed=processed,
                           skipped=skipped, failed=failed)

    return {"processed": processed, "skipped": skipped, "failed": failed,
            "run_id": run_id}
