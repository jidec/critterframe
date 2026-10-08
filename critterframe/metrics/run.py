"""run_metrics(), RunContext, and what a metric run counts as already done."""

import contextlib
import logging
from collections import Counter

from .. import drivers, segments as segment_iteration
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, hash_spec, load_json
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
    """What a run tells its operations' `prepare()` hooks before the per-occurrence loop.

    Args:
        project_path: Project being processed.
        occurrence_ids: Ids the run covers, after `subset` and `limit` but before
            already-completed work is skipped.
        part: Part being measured.
        run_name: Name of the run.
        report: The run's visualization report, for `context.report.figure()`. Falsy when
            visualization is off.
        reference: Whether the run measures reference masks.
        transforms: The run's transform chain.
    """

    def __init__(
        self,
        project_path,
        occurrence_ids,
        part,
        run_name,
        report=pipeline_visualization.NULL_REPORT,
        reference=False,
        transforms=(),
    ):
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
    """Return one metric value as short text for a panel caption."""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(value)[:3]) + "}"
    text = str(value)
    return text if len(text) <= 14 else text[:13] + "…"


def _reads_stored(operation):
    """Return whether a metric is computed from stored values instead of a segment."""
    return getattr(operation, "input", "segment") == "stored"


def _bind_same_run_features(metrics):
    """Return `metrics` with each stored-input metric bound to the ones listed before it.

    Raises:
        ValueError: If a metric reads a feature of its own run that isn't listed before it.
    """
    bound = []
    earlier = set()
    for operation in metrics:
        if _reads_stored(operation):
            operation = operation.bind(list(bound))
        if _reads_stored(operation) and getattr(operation, "from_run", "") is None:
            missing = [
                feature.metric_name
                for feature in operation.features
                if hash_spec(feature.spec()) not in earlier
            ]
            if missing:
                raise ValueError(
                    f"{operation.metric_name} reads {missing} from this run (from_run=None), "
                    "but no metric configured that way is listed before it in metrics= -- "
                    "list the same operations first, or pass from_run= to read another run"
                )
        earlier.add(hash_spec(operation.spec()))
        bound.append(operation)
    return bound


def _visualize_measurement(state, rows):
    """Emit the panel every metric run contributes: the measured segment with its values written on it."""
    if state.panel_sink is None:
        return

    panel = segment_panel(
        state.image,
        state.mask,
        lines=[f"{row['metric_name']}={_format_value(row['value'])}" for row in rows[:6]],
    )
    state.emit_panel(panel, "measured")


def _completed_keys(project_path, recipe_hash, source_mask_hashes=None, run_name=None):
    """Return the `(occurrence_id, part)` pairs a recipe already has values for.

    Args:
        project_path: Project to read from.
        recipe_hash: The recipe.
        source_mask_hashes: `{(occurrence_id, part): derivation hash}` of the masks about
            to be measured. A value then counts only if it was measured from that mask.
        run_name: Count only values written under this run name. None counts values
            written under any name, which is what a copy onto a new name reads.
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
        if row["source_mask_hash"] == source_mask_hashes.get((row["occurrence_id"], row["part"]), _NO_MASK)
    }


def _current_for_population(project_path, keys, recipe_hash, part, prepared):
    """Narrow keys to the ones whose stored values were fitted on this run's population.

    Args:
        project_path: Project to read from.
        keys: Candidate `(occurrence_id, part)` pairs, already matched on recipe hash.
        recipe_hash: The recipe.
        part: Part being measured.
        prepared: This run's `Recipe.prepare_all()` result. Empty for a recipe with no
            group metric, and then every key passes.
    """
    if not prepared or not keys:
        return keys

    # Filtered here and not with an IN list: one bound parameter per occurrence
    # passes SQLite's variable limit on a large project.
    occurrence_ids = {occurrence_id for occurrence_id, _part in keys}
    with run_records.open_database(project_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT occurrence_id, run_id FROM metrics
            WHERE recipe_hash = ? AND part = ?
            """,
            [recipe_hash, part],
        ).fetchall()
    run_id_by_occurrence = {
        row["occurrence_id"]: row["run_id"] for row in rows if row["occurrence_id"] in occurrence_ids
    }

    contexts = {}
    for run_id in set(run_id_by_occurrence.values()):
        found = run_records.load_runs(project_path, run_id=run_id)
        contexts[run_id] = found.iloc[0]["context"] if not found.empty else None

    def same_fit(stored, record):
        # The upstream recipe sits beside the population: a group metric fit
        # on values another recipe has since replaced is stale even when the
        # occurrences are the same ones.
        # fit_hash is for a fit that depends on more than WHO was in it: a
        # label-fitted score moves when a label changes on the same ids. Absent
        # from every other record, where None matches None.
        stored, record = stored or {}, record or {}
        return (
            stored.get("population", {}).get("ids_hash") == record.get("population", {}).get("ids_hash")
            and stored.get("from_recipe_hash") == record.get("from_recipe_hash")
            and stored.get("fit_hash") == record.get("fit_hash")
        )

    def population_matches(occurrence_id):
        run_id = run_id_by_occurrence.get(occurrence_id)
        context = contexts.get(run_id) or {}
        operations = context.get("operations") or {}
        return all(same_fit(operations.get(metric_name), record) for metric_name, record in prepared.items())

    return {key for key in keys if population_matches(key[0])}


def _copyable_rows(project_path, recipe_hash, keys, part):
    """Return existing values for keys, to insert under a new run instead of recomputing.

    Reads each occurrence's rows from its newest run only: one run writes an occurrence's
    values together.

    Returns:
        `{occurrence_id: [rows for make_metric_row]}`.
    """
    if not keys:
        return {}

    # Filtered in the loop and not with an IN list: one bound parameter per
    # occurrence passes SQLite's variable limit on a large project.
    occurrence_ids = {occurrence_id for occurrence_id, _part in keys}
    newest_run = {}
    batches = {}
    with run_records.open_database(project_path) as connection:
        rows = connection.execute(
            """
            SELECT metric_id, run_id, occurrence_id, metric_name, value_json,
                   unit, source_mask_hash
            FROM metrics
            WHERE recipe_hash = ? AND part = ?
            ORDER BY metric_id DESC
            """,
            [recipe_hash, part],
        )
        for row in rows:
            occurrence_id = row["occurrence_id"]
            if occurrence_id not in occurrence_ids:
                continue
            run_id = newest_run.setdefault(occurrence_id, row["run_id"])
            if row["run_id"] != run_id:
                continue
            batches.setdefault(occurrence_id, []).append(
                metric_records.make_metric_row(
                    occurrence_id,
                    part,
                    row["metric_name"],
                    load_json(row["value_json"]),
                    unit=row["unit"],
                    source_mask_hash=row["source_mask_hash"],
                )
            )

    return batches


def run_metrics(
    project_path,
    metrics,
    run_name=None,
    transforms=(),
    part=DEFAULT_PART,
    parts=None,
    subset=None,
    limit=None,
    force=False,
    visualize=True,
    visualize_every=None,
    reference=False,
    from_part=None,
    retry_failed=False,
):
    """Run a metric recipe over a project's occurrence-parts.

    Args:
        project_path: Project to process.
        metrics: Metric operations, evaluated in order. A `derived()` or `color_presence()`
            reading its own run takes the values of the metrics listed before it.
        run_name: Name for the run, which every export column and stored-value read uses.
            Defaults to the metric's name when there is exactly one; required otherwise.
        transforms: Ordered transforms applied before measuring. Any transform makes the
            recipe need a mask.
        part: Part to measure.
        parts: Several parts, each measured by the same recipe under its own run.
        subset: Named subset to process.
        limit: Cap on occurrences considered.
        force: Recompute occurrence-parts this recipe already covered, and allow `run_name`
            to move onto a different recipe. Resume an interrupted forced run without it.
        visualize: As in `run_segments`. The last grid column is the measured segment with
            its values.
        visualize_every: As in `run_segments`.
        reference: Measure the reference masks.
        from_part: Run `transforms` against this upstream part's canonical mask and measure
            `part`'s own mask in that frame, as `run_segments(from_part=)` produced it.
        retry_failed: Attempt occurrence-parts already recorded as failed under this recipe
            and mask.

    Returns:
        `{part: summary}`, each a `drivers.Tally.summary` plus `run_id`, `copied`
        (values reused from an identical recipe under another name), `previously_failed`
        and `elapsed_s`.

    Raises:
        ValueError: If `run_name` already points at a different recipe for the part and
            `force` is not set.
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
    clashes = set(segment_iteration.operation_labels(transforms)) & {
        operation.metric_name for operation in metrics
    }
    if clashes:
        raise ValueError(
            f"transform(s) {sorted(clashes)} share a name with a metric -- their "
            "recorded info would land in the same export column; pass name= to "
            "the metric"
        )

    # Before any recipe is built, so a recipe never changes once it exists.
    metrics = _bind_same_run_features(metrics)

    if transforms and all(_reads_stored(operation) for operation in metrics):
        raise ValueError(
            "transforms= has nothing to act on: every metric here reads stored "
            "values (input='stored'), so no segment is built"
        )

    target_parts = list(parts) if parts else [part]
    occurrence_ids = subset_selection.select_ids(project_path, subset=subset, limit=limit)
    logger.info(
        "run_metrics '%s': %d occurrence(s), part(s): %s",
        run_name,
        len(occurrence_ids),
        ", ".join(target_parts),
    )

    results = {}
    for target_part in target_parts:
        results[target_part] = _run_one_part(
            project_path,
            run_name,
            metrics,
            transforms,
            target_part,
            occurrence_ids,
            subset,
            limit,
            force,
            visualize,
            visualize_every,
            reference,
            from_part,
            retry_failed,
        )
    return results


def _run_one_part(
    project_path,
    run_name,
    metrics,
    transforms,
    part,
    occurrence_ids,
    subset,
    limit,
    force,
    visualize,
    visualize_every,
    reference,
    from_part=None,
    retry_failed=False,
):
    """Run the recipe for one part."""
    recipe = Recipe(
        "metric",
        run_name,
        transforms + metrics,
        part=part,
        from_part=from_part,
        inputs={"masks": "reference" if reference else "canonical"},
    )

    # Reference and canonical passes record failures under their own stage,
    # for the reason run_segments gives: records.failures keys on
    # (occurrence_id, part, stage), so sharing one would have each pass
    # overwrite and clear the other's rows.
    stage = "metric_reference" if reference else "metric"

    # Fails fast, before any mask lookup or prepare() hook runs, if run_name
    # already points at a different recipe for this part and force wasn't
    # given to move it. needs_currency_commit is only True for a forced move
    # still waiting on its first written value -- see commit_currency below.
    needs_currency_commit = run_records.resolve_recipe_currency(
        project_path, "metric", run_name, part, recipe.hash, force, recipe_spec=recipe.spec()
    )

    # Opened before prepare() so a group metric can write its fit figures; the
    # sample is fixed later, once begin() knows what this run will measure.
    report = pipeline_visualization.open_report(
        project_path,
        run_name,
        recipe.hash,
        part=part,
        visualize=visualize,
        visualize_every=visualize_every,
        identity=recipe.spec(),
    )

    prepared = recipe.prepare_all(
        RunContext(
            project_path,
            occurrence_ids,
            part,
            run_name,
            report=report,
            reference=reference,
            transforms=transforms,
        )
    )

    # A recipe needs a mask unless every metric in it says otherwise and
    # there's nothing transforming the segment first (a transform chain is
    # assumed to want a mask -- remove_appendages()/orient() and friends all
    # do). This only controls which occurrences the run REACHES (see todo
    # below) and whether staleness is judged by mask content -- masks are
    # looked up either way, so a maskless recipe still shows one in its panel
    # and records accurate provenance wherever one happens to exist.
    needs_mask = bool(transforms) or any(getattr(operation, "requires_mask", True) for operation in metrics)
    needs_segment = any(not _reads_stored(operation) for operation in metrics)

    mask_rows = mask_records.mask_lookup(
        project_path, part=part, occurrence_ids=occurrence_ids, reference=reference
    )

    # Completion is per (recipe, mask), not per recipe: a value already
    # computed by this recipe from a mask that has since been resegmented
    # describes a mask this run isn't measuring, so it can't stand in for
    # the work. The hashes come from the rows already loaded above, so this
    # costs no extra read. derivation_hash rather than the bare recipe hash
    # so that a part derived from another part is seen to change when its
    # upstream does, which the wing's own unchanged recipe hash would
    # otherwise hide.
    source_hashes = {
        (occurrence_id, part): mask_records.derivation_hash(row["recipe_hash"], row.get("source_mask_hash"))
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
        done_here = _completed_keys(
            project_path, recipe.hash, source_mask_hashes=staleness_hashes, run_name=run_name
        )
        done_here = _current_for_population(project_path, done_here, recipe.hash, part, prepared)

        done_elsewhere = (
            _completed_keys(project_path, recipe.hash, source_mask_hashes=staleness_hashes, run_name=None)
            - done_here
        )
        done_elsewhere = _current_for_population(project_path, done_elsewhere, recipe.hash, part, prepared)
        if done_elsewhere:
            copy_rows = _copyable_rows(project_path, recipe.hash, done_elsewhere, part)

    # What was attempted, for the failure ledger: this recipe over this
    # occurrence's current mask, so a retuned recipe or a resegmentation
    # retries by itself and only an unchanged repeat of the same attempt is
    # skipped. A maskless metric keys on the recipe alone, for the reason
    # staleness_hashes gives above -- its result doesn't depend on the mask.
    context_hashes = {
        (occurrence_id, part): mask_records.derivation_hash(
            recipe.hash, source_hashes.get((occurrence_id, part)) if needs_mask else None
        )
        for occurrence_id in occurrence_ids
    }
    failed_before = set()
    if not force and not retry_failed:
        failed_before = failure_records.failed_keys(project_path, stage, context_hashes)

    todo = [
        occurrence_id
        for occurrence_id in occurrence_ids
        if (not needs_mask or occurrence_id in mask_rows)
        and (occurrence_id, part) not in done_here
        and (occurrence_id, part) not in failed_before
        and occurrence_id not in copy_rows
    ]
    unsegmented = len([i for i in occurrence_ids if i not in mask_rows]) if needs_mask else 0
    skipped = len(done_here)
    tally = drivers.Tally(attempted=len(occurrence_ids))
    tally.skipped = skipped
    tally.no_input = unsegmented

    logger.info(
        "  %s: %d to measure, %d already done by recipe %s over the "
        "same mask, %d copied from another name's identical work, "
        "%d with no mask, %d previously failed (retry_failed=True to "
        "retry)",
        part,
        len(todo),
        skipped,
        recipe.hash,
        len(copy_rows),
        unsegmented,
        len(failed_before),
    )
    if unsegmented:
        logger.warning(
            "  %s: %d occurrence(s) have no '%s' mask -- run segmentation for that part first",
            part,
            unsegmented,
            part,
        )

    report.begin(todo)

    run_id = run_records.start_run(
        project_path,
        recipe,
        subset=subset,
        context={"occurrences": ids_record(occurrence_ids), "limit": limit, "operations": prepared},
    )

    # Copied first, in one batch write, before anything is measured for real --
    # cheap because it's a straight re-insert of values another run already
    # computed, no image, transform, or model involved.
    def commit_currency():
        # A forced move is committed at the FIRST value written rather than at
        # the end of the run: values are stored per occurrence, so a run
        # interrupted before an end-of-run commit would leave them stored under
        # a recipe the name doesn't point at -- not current, and not resumable,
        # since force is what gets past the name check and force also redoes
        # everything. A run that writes nothing still never moves the pointer.
        nonlocal needs_currency_commit
        if needs_currency_commit:
            run_records.commit_recipe_currency(project_path, "metric", run_name, part, recipe.hash)
            needs_currency_commit = False

    if copy_rows:
        metric_records.append_metrics(
            project_path, run_id, recipe.hash, [row for rows in copy_rows.values() for row in rows]
        )
        commit_currency()

    processed = 0
    failure_rows = []
    resolved = []
    missing = Counter()
    transform_labels = segment_iteration.operation_labels(transforms)
    source_rows = (
        mask_records.mask_lookup(project_path, part=from_part, occurrence_ids=todo)
        if from_part is not None
        else {}
    )

    progress = drivers.Progress(
        len(todo), f"run_metrics '{run_name}' part '{part}'", tallies=[tally], log=logger.info
    )

    # A recipe of stored-input metrics alone reads no image and builds no
    # Segment: what it is computed from is already in the metrics table.
    store = ImageStore(project_path, readonly=True) if needs_segment else contextlib.nullcontext()
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
                    raise drivers.NoInput(drivers.no_mask(from_part))
                from_mask = mask_records.decode_mask(source_row)

            state = segment_iteration.build_segment(
                image,
                mask=mask,
                occurrence_id=occurrence_id,
                part=part,
                project_path=project_path,
                panel_sink=report.sink(occurrence_id),
                from_mask=from_mask,
                from_part=from_part,
            )

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
                state, transform_info = build_state(occurrence_id) if needs_segment else (None, [])
                # In list order, so a derived metric with from_run=None reads
                # the values measured before it for this occurrence.
                computed = {}
                rows = []
                for operation in metrics:
                    value = operation(
                        StoredValues(occurrence_id, part, computed) if _reads_stored(operation) else state
                    )
                    computed[operation.metric_name] = value
                    rows.append(
                        metric_records.make_metric_row(
                            occurrence_id,
                            part,
                            operation.metric_name,
                            value,
                            unit=operation.unit,
                            source_mask_hash=source_hashes.get((occurrence_id, part)),
                        )
                    )

                if state is not None:
                    _visualize_measurement(state, rows)

                rows += [
                    metric_records.make_metric_row(
                        occurrence_id,
                        part,
                        label,
                        info,
                        unit=metric_records.TRANSFORM_INFO_UNIT,
                        source_mask_hash=source_hashes.get((occurrence_id, part)),
                    )
                    for label, info in transform_info
                    if info
                ]

                # Written per occurrence rather than buffered to the end: a
                # human-annotation run can be hours of clicking, and an
                # interruption should cost the current occurrence, not the day.
                metric_records.append_metrics(project_path, run_id, recipe.hash, rows)
                commit_currency()
                processed += 1
                tally.processed += 1
                resolved.append((occurrence_id, part))

            except drivers.NoInput as exc:
                tally.no_input += 1
                missing[str(exc)] += 1
            except Exception as exc:
                tally.record_failure(occurrence_id, exc)
                report.failure(occurrence_id, exc)
                logger.warning("metrics failed for %s part '%s': %s", occurrence_id, part, exc)
                failure_rows.append(
                    {
                        "occurrence_id": occurrence_id,
                        "part": part,
                        "context_hash": context_hashes[(occurrence_id, part)],
                        "error": exc,
                    }
                )

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
    run_records.finish_run(
        project_path,
        run_id,
        processed=processed + len(copy_rows),
        skipped=skipped,
        failed=tally.failed,
        flags=tally.flags,
    )

    if needs_currency_commit:
        logger.warning(
            "run_name '%s': force=True allowed a recipe change for part "
            "'%s', but nothing was processed -- it still points at the "
            "previous recipe",
            run_name,
            part,
        )

    return tally.summary(
        copied=len(copy_rows), run_id=run_id, previously_failed=len(failed_before), elapsed_s=elapsed
    )
