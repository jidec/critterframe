"""
run_metrics() + RunContext + _completed_keys.

Transforms shape what gets measured and are part of the recipe hash, but the
segment they produce is thrown away -- only the values are kept, with each
transform's scalar info stored beside them as a `transform_info` row. Occurrences with
no mask for the part are neither measured nor counted done, and the run says so.
"""

import contextlib
import logging
from collections import Counter

from .. import drivers, segments as segment_iteration
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, load_json
from ..records import failures as failure_records
from ..records import masks as mask_records
from ..records import metrics as metric_records
from ..records import runs as run_records
from ..records.occurrences import ids_record
from ..storage.imagestore import ImageStore
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import segment_panel
from .stored import StoredValues

logger = logging.getLogger(__name__)


class RunContext:
    """
    What a run tells its operations before the per-occurrence loop starts.

    Passed to every operation's prepare() hook. Almost all of them ignore it --
    a metric is normally self-contained per occurrence -- but group metrics need
    to know which project and which occurrences make up the reference population
    they're about to fit against, and this is how they find out (see
    metrics.outliers).

    - `project_path` -- project being processed.
    - `occurrence_ids` -- ids this run covers, AFTER subset and limit are
      applied but BEFORE skipping already-completed work. A group metric must
      fit against the whole population it's scoring within, not against
      whichever occurrences happen to be left over from an interrupted earlier
      run -- otherwise resuming a run would silently change what "outlier"
      means partway through it.
    - `part` -- part being measured.
    - `run_name` -- name of the run.
    - `report` -- the run's visualization report, so prepare() can write a
      whole-population figure with `context.report.figure()`. A NullReport when
      visualization is off; build an expensive figure only `if
      context.report:`.
    - `reference` -- whether this run measures reference masks rather than
      canonical ones.
    - `transforms` -- the run's transform chain, so prepare() can draw
      segments the way the run sees them.
    """

    def __init__(self, project_path, occurrence_ids, part, run_name,
                 report=pipeline_visualization.NULL_REPORT, reference=False,
                 transforms=()):
        self.project_path = project_path
        self.occurrence_ids = list(occurrence_ids)
        self.part = part
        self.run_name = run_name
        self.report = report
        self.reference = reference
        self.transforms = list(transforms)


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


def _reads_stored(operation):
    """Whether a metric is computed from stored values (a StoredValues) rather than a Segment."""
    return getattr(operation, "input", "segment") == "stored"


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

    panel = segment_panel(state.image, state.mask, lines=[
        f"{row['metric_name']}={_format_value(row['value'])}" for row in rows[:6]])
    state.emit_panel(panel, "measured")


def _completed_keys(project_path, recipe_hash, source_mask_hashes=None, run_name=None):
    """
    The (occurrence_id, part) pairs a recipe has already produced values for --
    the repeat-awareness check this module makes before doing any work.

    Here rather than in records.metrics because it is this run's decision about
    what work is left, not a way of reading the metric log. Keyed on the recipe
    hash and not the run, so work completed by an earlier interrupted run of the
    same recipe is exactly what a new run skips.

    - `source_mask_hashes` -- {(occurrence_id, part): derivation_hash} of the
      masks about to be measured. When given, completion means "this recipe has
      run over THIS mask" rather than "this recipe has run here", so
      resegmenting forces a recompute. Rows the caller has no mask for never
      count. Omit it to ask the weaker question, which is all a caller with no
      masks in hand can ask.
    - `run_name` -- restrict to rows written by a run under this exact name.
      recipe_hash no longer encodes name (see Recipe.hash), so without this,
      "already done" would mean "done by ANY name" -- which is exactly what
      _copyable_rows below wants, but not what "does THIS name already have its
      own values" should mean: a run_name is the key export.column_name and
      records.metrics.latest_values read values back by, so a brand-new name
      must actually get its own rows, never silently borrow another name's
      completion. None asks the weaker "has this exact recipe run at all, under
      any name" question.
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

    - `keys` -- candidate (occurrence_id, part) pairs, already matched on
      recipe_hash (and, if the caller checked it, source_mask_hash).
    - `prepared` -- THIS run's own recipe.prepare_all(context) result, keyed by
      metric_name. Empty for a recipe with no group metric, in which case every
      key passes untouched -- an ordinary metric's value never depends on who
      else was in scope.
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

    def same_fit(stored, record):
        # The upstream recipe sits beside the population: a group metric fit
        # on values another recipe has since replaced is stale even when the
        # occurrences are the same ones.
        stored, record = stored or {}, record or {}
        return (stored.get("population", {}).get("ids_hash")
                == record.get("population", {}).get("ids_hash")
                and stored.get("from_recipe_hash") == record.get("from_recipe_hash"))

    def population_matches(occurrence_id):
        run_id = run_id_by_occurrence.get(occurrence_id)
        context = contexts.get(run_id) or {}
        operations = context.get("operations") or {}
        return all(same_fit(operations.get(metric_name), record)
                   for metric_name, record in prepared.items())

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
                reference=False, from_part=None, retry_failed=False):
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
    - `retry_failed` -- attempt occurrence-parts that already failed under
      this recipe and this mask. Normally skipped, the same way
      `run_segments` skips them: a metric that raised on an occurrence raises
      again on a rerun, and a long annotation or embedding pass shouldn't
      spend the attempt finding that out. A recipe change or a resegmentation
      retries by itself, since the recorded failure is keyed on both.
    - `from_part` -- run `transforms` against an upstream part's CANONICAL
      mask and measure `part`'s own mask inside the resulting frame, the same
      way `run_segments(from_part=...)` produced it (see
      `segments.iterate_segments`). Pass the same `from_part` that run used,
      or a part carved out of the organism crop is measured in a crop of its
      own instead of the one it was segmented in.

    Returns {part: summary}, each as `drivers.Tally.summary` plus `run_id`,
    `copied` -- occurrence-parts whose value was re-used from an identically
    configured run under a different name rather than recomputed (see
    `run_name` above) -- `previously_failed`, and `elapsed_s`. `no_input` counts
    occurrence-parts with no mask, image or `from_part` mask yet; they are
    attempted again once it exists.
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

    # Each transform's info is stored as a metric named by its label, so the
    # two must never share a name, or one column would hold both.
    clashes = (set(segment_iteration.operation_labels(transforms))
               & {operation.metric_name for operation in metrics})
    if clashes:
        raise ValueError(
            f"transform(s) {sorted(clashes)} share a name with a metric -- their "
            "recorded info would land in the same export column; pass name= to "
            "the metric")

    if transforms and all(_reads_stored(operation) for operation in metrics):
        raise ValueError(
            "transforms= has nothing to act on: every metric here reads stored "
            "values (input='stored'), so no segment is built")

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
            reference, from_part, retry_failed,
        )
    return results


def _run_one_part(project_path, run_name, metrics, transforms, part,
                  occurrence_ids, subset, limit, force, visualize,
                  visualize_every, reference, from_part=None,
                  retry_failed=False):
    """Execute one part's recipe. Split out so the multi-part loop stays readable."""
    recipe = Recipe("metric", run_name, transforms + metrics, part=part,
                    from_part=from_part,
                    inputs={"masks": "reference" if reference else "canonical"})

    # Reference and canonical passes record failures under their own stage,
    # for the reason run_segments gives: records.failures keys on
    # (occurrence_id, part, stage), so sharing one would have each pass
    # overwrite and clear the other's rows.
    stage = "metric_reference" if reference else "metric"

    # Fails fast, before any mask lookup or prepare() hook runs, if run_name
    # already points at a different recipe for this part and force wasn't
    # given to move it. needs_currency_commit is only True for a forced move
    # still waiting on confirmation that it produced something -- see below.
    needs_currency_commit = run_records.resolve_recipe_currency(
        project_path, "metric", run_name, part, recipe.hash, force)

    # Opened before prepare() so a group metric can write its fit figures; the
    # sample is fixed later, once begin() knows what this run will measure.
    report = pipeline_visualization.open_report(
        project_path, run_name, recipe.hash, part=part, visualize=visualize,
        visualize_every=visualize_every, identity=recipe.spec())

    prepared = recipe.prepare_all(
        RunContext(project_path, occurrence_ids, part, run_name, report=report,
                   reference=reference, transforms=transforms))

    # A recipe needs a mask unless every metric in it says otherwise and
    # there's nothing transforming the segment first (a transform chain is
    # assumed to want a mask -- remove_appendages()/orient() and friends all
    # do). This only controls which occurrences the run REACHES (see todo
    # below) and whether staleness is judged by mask content -- masks are
    # looked up either way, so a maskless recipe still shows one in its panel
    # and records accurate provenance wherever one happens to exist.
    needs_mask = bool(transforms) or any(
        getattr(operation, "requires_mask", True) for operation in metrics)
    needs_segment = any(not _reads_stored(operation) for operation in metrics)

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

    # What was attempted, for the failure ledger: this recipe over this
    # occurrence's current mask, so a retuned recipe or a resegmentation
    # retries by itself and only an unchanged repeat of the same attempt is
    # skipped. A maskless metric keys on the recipe alone, for the reason
    # staleness_hashes gives above -- its result doesn't depend on the mask.
    context_hashes = {
        (occurrence_id, part): mask_records.derivation_hash(
            recipe.hash,
            source_hashes.get((occurrence_id, part)) if needs_mask else None)
        for occurrence_id in occurrence_ids
    }
    failed_before = set()
    if not force and not retry_failed:
        failed_before = failure_records.failed_keys(project_path, stage,
                                                    context_hashes)

    todo = [
        occurrence_id for occurrence_id in occurrence_ids
        if (not needs_mask or occurrence_id in mask_rows)
        and (occurrence_id, part) not in done_here
        and (occurrence_id, part) not in failed_before
        and occurrence_id not in copy_rows
    ]
    unsegmented = (len([i for i in occurrence_ids if i not in mask_rows])
                  if needs_mask else 0)
    skipped = len(done_here)
    tally = drivers.Tally(attempted=len(occurrence_ids))
    tally.skipped = skipped
    tally.no_input = unsegmented

    logger.info("  %s: %d to measure, %d already done by recipe %s over the "
                "same mask, %d copied from another name's identical work, "
                "%d with no mask, %d previously failed (retry_failed=True to "
                "retry)", part, len(todo), skipped, recipe.hash,
                len(copy_rows), unsegmented, len(failed_before))
    if unsegmented:
        logger.warning("  %s: %d occurrence(s) have no '%s' mask -- run "
                       "segmentation for that part first", part, unsegmented, part)

    report.begin(todo)

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
    failure_rows = []
    resolved = []
    missing = Counter()
    transform_labels = segment_iteration.operation_labels(transforms)
    source_rows = (mask_records.mask_lookup(project_path, part=from_part,
                                            occurrence_ids=todo)
                   if from_part is not None else {})

    progress = drivers.Progress(
        len(todo), f"run_metrics '{run_name}' part '{part}'", tallies=[tally],
        log=logger.info)

    # A recipe of stored-input metrics alone reads no image and builds no
    # Segment: what it is computed from is already in the metrics table.
    store = (ImageStore(project_path, readonly=True) if needs_segment
             else contextlib.nullcontext())
    with store as images:

        def build_state(occurrence_id):
            image = images.get(occurrence_id)
            if image is None:
                raise drivers.NoInput(drivers.NO_IMAGE)

            # .get(), not [] -- absent for a maskless recipe (mask_rows is
            # {} entirely) and, defensively, for any occurrence a
            # mask-requiring recipe's own todo filter already excluded.
            mask_row = mask_rows.get(occurrence_id)
            mask = mask_records.decode_mask(mask_row) if mask_row is not None else None

            from_mask = None
            if from_part is not None:
                source_row = source_rows.get(occurrence_id)
                if source_row is None:
                    raise drivers.NoInput(
                        drivers.no_mask(from_part))
                from_mask = mask_records.decode_mask(source_row)

            state = segment_iteration.build_segment(
                image, mask=mask, occurrence_id=occurrence_id, part=part,
                project_path=project_path,
                panel_sink=report.sink(occurrence_id),
                from_mask=from_mask, from_part=from_part)

            transform_info = []
            for label, operation in zip(transform_labels, transforms):
                state, info = operation(state)
                tally.record_flags(info)
                transform_info.append((label, segment_iteration.scalar_info(info)))

            if from_mask is not None:
                state = state.for_part(part)
                state.mask = None if mask is None else state.project_mask(mask)
            return state, transform_info

        for occurrence_id in todo:
            try:
                state, transform_info = (build_state(occurrence_id) if needs_segment
                                         else (None, []))
                stored = StoredValues(occurrence_id, part)

                rows = [
                    metric_records.make_metric_row(
                        occurrence_id, part, operation.metric_name,
                        operation(stored if _reads_stored(operation) else state),
                        unit=operation.unit,
                        source_mask_hash=source_hashes.get((occurrence_id, part)),
                    )
                    for operation in metrics
                ]

                if state is not None:
                    _visualize_measurement(state, rows)

                rows += [
                    metric_records.make_metric_row(
                        occurrence_id, part, label, info,
                        unit=metric_records.TRANSFORM_INFO_UNIT,
                        source_mask_hash=source_hashes.get((occurrence_id, part)),
                    )
                    for label, info in transform_info if info
                ]

                # Written per occurrence rather than buffered to the end: a
                # human-annotation run can be hours of clicking, and an
                # interruption should cost the current occurrence, not the day.
                metric_records.append_metrics(project_path, run_id, recipe.hash, rows)
                processed += 1
                tally.processed += 1
                resolved.append((occurrence_id, part))

            except drivers.NoInput as exc:
                tally.no_input += 1
                missing[str(exc)] += 1
            except Exception as exc:
                tally.record_failure(occurrence_id, exc)
                report.failure(occurrence_id, exc)
                logger.warning("metrics failed for %s part '%s': %s",
                               occurrence_id, part, exc)
                failure_rows.append({
                    "occurrence_id": occurrence_id,
                    "part": part,
                    "context_hash": context_hashes[(occurrence_id, part)],
                    "error": exc,
                })

            report.done(occurrence_id)
            progress.step()

    report.close()
    elapsed = progress.finish()
    drivers.log_no_input(missing, f"run_metrics part '{part}'")

    # Written once at the end rather than per occurrence: unlike the metric
    # rows above, a lost failure costs a retry, not a day's annotation.
    failure_records.record_failures(project_path, stage, failure_rows)
    failure_records.clear_failures(project_path, stage, resolved)

    # Copied rows count toward the stored n_processed: real rows were newly
    # attributed to this run_id, unlike a skip, which wrote nothing at all.
    run_records.finish_run(project_path, run_id,
                           processed=processed + len(copy_rows),
                           skipped=skipped, failed=tally.failed, flags=tally.flags)

    if needs_currency_commit:
        if processed > 0 or copy_rows:
            run_records.commit_recipe_currency(project_path, "metric",
                                               run_name, part, recipe.hash)
        else:
            logger.warning(
                "run_name '%s': force=True allowed a recipe change for part "
                "'%s', but nothing was processed -- it still points at the "
                "previous recipe", run_name, part)

    return tally.summary(copied=len(copy_rows), run_id=run_id,
                         previously_failed=len(failed_before), elapsed_s=elapsed)
