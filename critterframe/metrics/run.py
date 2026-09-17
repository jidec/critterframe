"""
run_metrics() + RunContext + _completed_keys.

Transforms shape what gets measured and are part of the recipe hash, but the
segment they produce is thrown away -- only the value is kept. Occurrences with
no mask for the part are neither measured nor counted done, and the run says so.
"""

import logging

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment, load_json
from ..records import masks as mask_records
from ..records import metrics as metric_records
from ..records import runs as run_records
from ..records.occurrences import ids_record
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


def _completed_keys(project_path, recipe_hash, source_mask_hashes=None, run_name=None):
    """
    The (occurrence_id, part) pairs a recipe has already produced values for --
    the repeat-awareness check this module makes before doing any work.

    Here rather than in records.metrics because it is this run's decision about
    what work is left, not a way of reading the metric log. Keyed on the recipe
    hash and not the run, so work completed by an earlier interrupted run of the
    same recipe is exactly what a new run skips.

    source_mask_hashes -- {(occurrence_id, part): derivation_hash} of the masks
                          about to be measured. When given, completion means
                          "this recipe has run over THIS mask" rather than "this
                          recipe has run here", so resegmenting forces a
                          recompute. Rows the caller has no mask for never count.
                          Omit it to ask the weaker question, which is all a
                          caller with no masks in hand can ask.
    run_name           -- restrict to rows written by a run under this exact
                          name. recipe_hash no longer encodes name (see
                          Recipe.hash), so without this, "already done" would
                          mean "done by ANY name" -- which is exactly what
                          _copyable_rows below wants, but not what "does THIS
                          name already have its own values" should mean: a
                          run_name is the key export.column_name and
                          records.metrics.latest_values read values back by,
                          so a brand-new name must actually get its own rows,
                          never silently borrow another name's completion.
                          None asks the weaker "has this exact recipe run at
                          all, under any name" question.
    """
    query = "SELECT DISTINCT m.occurrence_id, m.part, m.source_mask_hash FROM metrics m"
    params = [recipe_hash]
    if run_name is not None:
        query += " JOIN runs r ON m.run_id = r.run_id WHERE m.recipe_hash = ? AND r.name = ?"
        params.append(run_name)
    else:
        query += " WHERE m.recipe_hash = ?"

    with run_records.open_database(project_path) as connection:
        rows = connection.execute(query, params).fetchall()

    if source_mask_hashes is None:
        return {(row["occurrence_id"], row["part"]) for row in rows}

    return {
        (row["occurrence_id"], row["part"])
        for row in rows
        if row["source_mask_hash"] == source_mask_hashes.get(
            (row["occurrence_id"], row["part"]), _NO_MASK)
    }


def _current_for_population(project_path, keys, recipe_hash, part, prepared):
    """
    Narrow `keys` to the ones whose stored values were fit against the SAME
    reference population `prepared` describes -- the check that makes a stored
    group-metric value trustworthy as "already done" or "safe to copy" across
    runs sharing one recipe_hash.

    A group metric's prepare() fits against context.occurrence_ids (this run's
    own subset/limit), which is deliberately not part of the hash -- for the
    same reason `subset` never is, so a growing project doesn't force constant
    recomputation. But unlike every other operation, a group metric's VALUE
    genuinely depends on that population, so two runs sharing a hash can
    disagree. prepare() already returns the fitted population as an
    ids_record and Recipe.prepare_all stores it in the run's context_json, so
    this costs one read of the (small) runs table, not a new column.

    keys     -- candidate (occurrence_id, part) pairs, already matched on
                recipe_hash (and, if the caller checked it, source_mask_hash).
    prepared -- THIS run's own recipe.prepare_all(context) result, keyed by
                metric_name. Empty for a recipe with no group metric, in which
                case every key passes untouched -- an ordinary metric's value
                never depends on who else was in scope.
    """
    if not prepared or not keys:
        return keys

    occurrence_ids = {occurrence_id for occurrence_id, _part in keys}
    placeholders = ",".join("?" for _ in occurrence_ids)
    with run_records.open_database(project_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT occurrence_id, run_id FROM metrics
            WHERE recipe_hash = ? AND part = ? AND occurrence_id IN ({placeholders})
            """,
            [recipe_hash, part, *occurrence_ids],
        ).fetchall()
    run_id_by_occurrence = {row["occurrence_id"]: row["run_id"] for row in rows}

    contexts = {}
    for run_id in set(run_id_by_occurrence.values()):
        found = run_records.load_runs(project_path, run_id=run_id)
        contexts[run_id] = found.iloc[0]["context"] if not found.empty else None

    def population_matches(occurrence_id):
        run_id = run_id_by_occurrence.get(occurrence_id)
        context = contexts.get(run_id) or {}
        operations = context.get("operations") or {}
        return all(
            (operations.get(metric_name) or {}).get("population", {}).get("ids_hash")
            == (record or {}).get("population", {}).get("ids_hash")
            for metric_name, record in prepared.items()
        )

    return {key for key in keys if population_matches(key[0])}


def _copyable_rows(project_path, recipe_hash, keys, part):
    """
    Existing stored values for `keys`, ready to be re-inserted under a new
    run_id instead of recomputed -- the cheap half of a rename. Only called for
    (occurrence_id, part) pairs already established elsewhere as safe: matching
    recipe_hash, matching source_mask_hash, and (via _current_for_population)
    matching reference population where that applies.

    Reads at most one row set per occurrence -- the NEWEST run's, since an
    occurrence's values under one recipe_hash are always written together in
    one run (see _run_one_part), never split across two. Ordered by metric_id
    DESC so the first row seen per occurrence fixes that occurrence's newest
    run_id; only later rows from that SAME run_id are folded in -- a plain
    "seen this occurrence before" check would wrongly keep just one of that
    run's several metric_name rows.

    Returns {occurrence_id: [rows for make_metric_row]}.
    """
    if not keys:
        return {}

    occurrence_ids = {occurrence_id for occurrence_id, _part in keys}
    placeholders = ",".join("?" for _ in occurrence_ids)
    with run_records.open_database(project_path) as connection:
        rows = connection.execute(
            f"""
            SELECT metric_id, run_id, occurrence_id, metric_name, value_json,
                   unit, source_mask_hash
            FROM metrics
            WHERE recipe_hash = ? AND part = ? AND occurrence_id IN ({placeholders})
            ORDER BY metric_id DESC
            """,
            [recipe_hash, part, *occurrence_ids],
        ).fetchall()

    newest_run = {}
    batches = {}
    for row in rows:
        occurrence_id = row["occurrence_id"]
        if occurrence_id not in occurrence_ids:
            continue
        run_id = newest_run.setdefault(occurrence_id, row["run_id"])
        if row["run_id"] != run_id:
            continue
        batches.setdefault(occurrence_id, []).append(
            metric_records.make_metric_row(
                occurrence_id, part, row["metric_name"], load_json(row["value_json"]),
                unit=row["unit"], source_mask_hash=row["source_mask_hash"]))

    return batches


def run_metrics(project_path, metrics, run_name=None, transforms=(),
                part=DEFAULT_PART, parts=None, subset=None, limit=None,
                force=False, visualize=True, visualize_every=None,
                reference=False):
    """
    Run a metric recipe over a project's occurrence-parts.

    - `project_path` -- the project to process.
    - `metrics` -- list of Metric operations to evaluate. Needs a mask unless
      every metric declares `requires_mask=False` (see Metric) AND
      `transforms` is empty -- a maskless recipe reaches every occurrence
      with an image, segmented or not, and measures with a maskless Segment;
      otherwise, exactly as before, only occurrence-parts already segmented
      are measured.
    - `run_name` -- what to call this run. Recorded on the run, not part of
      recipe identity (see Recipe.hash) -- but still what every exported
      column and `records.metrics.latest_values` read values back by, so two
      differently-configured measurements of one trait stay distinguishable,
      and a rename never silently reads as "already done" the way it does
      for `run_segments` (nothing there is keyed by name). Renaming here
      costs a cheap copy instead: if this exact recipe already has values
      elsewhere, they're copied onto this run rather than recomputed -- see
      the `copied` count below.
      Defaults to the metric's own `metric_name` when `metrics` holds
      exactly one -- the common case for a single screening pass
      (`metrics=[cf.usability_annotation()]`), where that name is the
      obviously right one and typing it twice is pure repetition. Measuring
      more than one metric in a call has no such obvious name, so it's
      required there: run_metrics raises immediately rather than guessing
      one, since guessing wrong would mean silently renaming an export
      column the moment a second metric gets added to an existing call.
    - `transforms` -- ordered Transform operations applied before measuring.
      Presence alone means the recipe needs a mask, whether or not every
      metric in it does -- a transform chain (remove_appendages(),
      orient(), ...) is assumed to want one.
    - `part` -- part to measure; the whole organism by default.
    - `parts` -- several parts, each with the same recipe. Each gets its own
      run record and recipe hash, so re-running one leaves the others alone.
    - `subset` -- name of a subset to process, or None for every occurrence.
    - `limit` -- optional cap on occurrences.
    - `force` -- two related meanings. Recompute occurrence-parts this recipe
      already covered from the mask it is about to measure (occurrences
      whose mask has been replaced are recomputed regardless: their stored
      value describes a mask that no longer exists). Also acknowledges
      moving `run_name` onto a genuinely different recipe for this part:
      without it, changing what `run_name` measures raises rather than
      silently taking over the name (`records.runs.resolve_recipe_currency`)
      -- values already on record stay on record but stop being current
      once this run has actually produced something under the new recipe.
    - `visualize` -- as in `run_segments` (default True). The last column of
      a metric grid is the measured segment with its values written on it.
    - `visualize_every` -- as in `run_segments`: in addition to the one grid
      above, write a second, independent series of checkpoint grids every N
      occurrences processed, each resampled fresh from that checkpoint's own
      window so it always has real content. None (default) writes none of
      these. No effect without `visualize`.
    - `reference` -- measure the reference masks instead of the canonical
      ones. How reference metric VALUES are produced for validation: the
      same recipe pointed at the reference table, so disagreement is
      attributable to the masks rather than the method.

    Returns {part: {"processed", "copied", "skipped", "failed", "run_id"}}.
    `copied` is occurrence-parts whose value was re-used from an identically
    configured run under a different name rather than recomputed -- see
    `run_name` above.
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

    # Resolved after the type checks above, so a wrong-kind operation raises
    # its own clear TypeError rather than an AttributeError from reaching for
    # .metric_name on something that isn't a Metric.
    if run_name is None:
        if len(metrics) != 1:
            raise ValueError(
                "run_metrics needs an explicit run_name when measuring more "
                "than one metric in one call -- there's no single obvious "
                "name for the result"
            )
        run_name = metrics[0].metric_name

    if visualize_every and not visualize:
        logger.warning(
            "visualize_every=%d with visualize=%r: there's no report to "
            "checkpoint without visualize, so visualize_every has no effect",
            visualize_every, visualize)

    target_parts = list(parts) if parts else [part]
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                 limit=limit)
    logger.info("run_metrics '%s': %d occurrence(s), part(s): %s",
                run_name, len(occurrence_ids), ", ".join(target_parts))

    results = {}
    for target_part in target_parts:
        results[target_part] = _run_one_part(
            project_path, run_name, metrics, transforms, target_part,
            occurrence_ids, subset, limit, force, visualize, visualize_every,
            reference,
        )
    return results


def _run_one_part(project_path, run_name, metrics, transforms, part,
                  occurrence_ids, subset, limit, force, visualize,
                  visualize_every, reference):
    """Execute one part's recipe. Split out so the multi-part loop stays readable."""
    recipe = Recipe("metric", run_name, transforms + metrics, part=part,
                    inputs={"masks": "reference" if reference else "canonical"})

    # Fails fast, before any mask lookup or prepare() hook runs, if run_name
    # already points at a different recipe for this part and force wasn't
    # given to move it. needs_currency_commit is only True for a forced move
    # still waiting on confirmation that it produced something -- see below.
    needs_currency_commit = run_records.resolve_recipe_currency(
        project_path, "metric", run_name, part, recipe.hash, force)

    prepared = recipe.prepare_all(
        RunContext(project_path, occurrence_ids, part, run_name))

    # A recipe needs a mask unless every metric in it says otherwise and
    # there's nothing transforming the segment first (a transform chain is
    # assumed to want a mask -- remove_appendages()/orient() and friends all
    # do). This only controls which occurrences the run REACHES (see todo
    # below) and whether staleness is judged by mask content -- masks are
    # looked up either way, so a maskless recipe still shows one in its panel
    # and records accurate provenance wherever one happens to exist.
    needs_mask = bool(transforms) or any(
        getattr(operation, "requires_mask", True) for operation in metrics)

    mask_rows = mask_records.mask_lookup(project_path, part=part,
                                         occurrence_ids=occurrence_ids,
                                         reference=reference)

    # Completion is per (recipe, mask), not per recipe: a value already
    # computed by this recipe from a mask that has since been resegmented
    # describes a mask this run isn't measuring, so it can't stand in for
    # the work. The hashes come from the rows already loaded above, so this
    # costs no extra read. derivation_hash rather than the bare recipe hash
    # so that a part derived from another part is seen to change when its
    # upstream does, which the wing's own unchanged recipe hash would
    # otherwise hide.
    source_hashes = {
        (occurrence_id, part): mask_records.derivation_hash(
            row["recipe_hash"], row.get("source_mask_hash"))
        for occurrence_id, row in mask_rows.items()
    }
    # A maskless metric's OUTPUT doesn't depend on mask content (it's a
    # judgement about the image, not the boundary), so resegmenting must not
    # make its already-recorded values look stale. None (not source_hashes),
    # passed below, asks completed_keys the weaker "has this recipe run for
    # this occurrence at all" question instead of comparing mask identity.
    staleness_hashes = source_hashes if needs_mask else None

    # "Already done" now means three things, not one, now that name isn't part
    # of the hash: done under THIS name (skip outright), done under some OTHER
    # name (cheap-copy its values instead of recomputing), or not done at all
    # (measure it). A group metric's fit depends on which occurrences were in
    # scope, so both sets are additionally narrowed to rows fit against the
    # SAME population this run just fit -- see _current_for_population.
    done_here = set()
    copy_rows = {}
    if not force:
        done_here = _completed_keys(project_path, recipe.hash,
                                    source_mask_hashes=staleness_hashes,
                                    run_name=run_name)
        done_here = _current_for_population(project_path, done_here,
                                            recipe.hash, part, prepared)

        done_elsewhere = _completed_keys(project_path, recipe.hash,
                                         source_mask_hashes=staleness_hashes,
                                         run_name=None) - done_here
        done_elsewhere = _current_for_population(project_path, done_elsewhere,
                                                  recipe.hash, part, prepared)
        if done_elsewhere:
            copy_rows = _copyable_rows(project_path, recipe.hash,
                                       done_elsewhere, part)

    todo = [
        occurrence_id for occurrence_id in occurrence_ids
        if (not needs_mask or occurrence_id in mask_rows)
        and (occurrence_id, part) not in done_here
        and occurrence_id not in copy_rows
    ]
    unsegmented = (len([i for i in occurrence_ids if i not in mask_rows])
                  if needs_mask else 0)
    skipped = len(done_here)

    logger.info("  %s: %d to measure, %d already done by recipe %s over the "
                "same mask, %d copied from another name's identical work, "
                "%d with no mask", part, len(todo), skipped, recipe.hash,
                len(copy_rows), unsegmented)
    if unsegmented:
        logger.warning("  %s: %d occurrence(s) have no '%s' mask -- run "
                       "segmentation for that part first", part, unsegmented, part)

    report = pipeline_visualization.run_report(project_path, run_name,
                                               recipe.hash, part, todo,
                                               visualize)

    # visualize_every's checkpoint series: a second RunReport, resampled
    # fresh each window from whatever that window actually processes rather
    # than the fixed whole-run sample above, so a checkpoint always has real
    # content -- see pipeline_visualization's module docstring.
    window_report = None
    combined_sink = report

    def rotate_window(start_index):
        nonlocal window_report, combined_sink
        if not (visualize_every and visualize) or start_index >= len(todo):
            return
        end = min(start_index + visualize_every, len(todo))
        sample = pipeline_visualization.resolve_sample(
            todo[start_index:end], visualize)
        window_report = (
            pipeline_visualization.RunReport(
                project_path, run_name, recipe.hash, part, sample,
                suffix=f"__at{end:08d}")
            if sample else None)
        combined = [existing for existing in (report, window_report)
                   if existing is not None]
        combined_sink = (
            pipeline_visualization.PanelFanout(combined) if len(combined) > 1
            else (combined[0] if combined else None))

    rotate_window(0)

    run_id = run_records.start_run(
        project_path, recipe, subset=subset,
        context={"occurrences": ids_record(occurrence_ids), "limit": limit,
                 "operations": prepared})

    # Copied first, in one batch write, before anything is measured for real --
    # cheap because it's a straight re-insert of values another run already
    # computed, no image, transform, or model involved.
    if copy_rows:
        metric_records.append_metrics(
            project_path, run_id, recipe.hash,
            [row for rows in copy_rows.values() for row in rows])

    processed = 0
    failed = 0

    with ImageStore(project_path, readonly=True) as images:
        for index, occurrence_id in enumerate(todo):
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                # .get(), not [] -- absent for a maskless recipe (mask_rows is
                # {} entirely) and, defensively, for any occurrence a
                # mask-requiring recipe's own todo filter already excluded.
                mask_row = mask_rows.get(occurrence_id)
                mask = mask_records.decode_mask(mask_row) if mask_row is not None else None
                state = Segment(image, mask=mask,
                                occurrence_id=occurrence_id, part=part,
                                project_path=project_path,
                                panel_sink=pipeline_visualization.panel_sink(
                                    combined_sink, occurrence_id))

                for operation in transforms:
                    state, _info = operation(state)

                rows = [
                    metric_records.make_metric_row(
                        occurrence_id, part, operation.metric_name,
                        operation(state), unit=operation.unit,
                        source_mask_hash=source_hashes.get((occurrence_id, part)),
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

            # Cheap even when nothing new was collected -- RunReport.save() is
            # a no-op then -- so this runs on every occurrence regardless of
            # whether it was itself in the sample.
            if visualize_every and (index + 1) % visualize_every == 0:
                if report is not None:
                    report.save()
                if window_report is not None:
                    window_report.save()
                rotate_window(index + 1)

    if report is not None:
        report.save()
    # A run that completes without landing exactly on a checkpoint boundary
    # still leaves a grid for its trailing partial window.
    if window_report is not None:
        window_report.save()

    # Copied rows count toward the stored n_processed: real rows were newly
    # attributed to this run_id, unlike a skip, which wrote nothing at all.
    run_records.finish_run(project_path, run_id,
                           processed=processed + len(copy_rows),
                           skipped=skipped, failed=failed)

    if needs_currency_commit:
        if processed > 0 or copy_rows:
            run_records.commit_recipe_currency(project_path, "metric",
                                               run_name, part, recipe.hash)
        else:
            logger.warning(
                "run_name '%s': force=True allowed a recipe change for part "
                "'%s', but nothing was processed -- it still points at the "
                "previous recipe", run_name, part)

    return {"processed": processed, "copied": len(copy_rows), "skipped": skipped,
            "failed": failed, "run_id": run_id}
