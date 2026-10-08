"""iterate_segments(): the per-occurrence loop most drivers walk, plus build_segment and how an operation's info is stored."""

import logging
from collections import Counter

from .drivers import NO_IMAGE, NoInput, Progress, Tally, log_no_input, no_mask
from .project import paths, subsets as subset_selection
from .recipes import DEFAULT_PART, Segment
from .records import masks as mask_records
from .storage.imagestore import ImageStore
from .visualization import pipeline as pipeline_visualization

logger = logging.getLogger(__name__)

_SCALARS = (bool, int, float, str, type(None))


def scalar_info(info):
    """Return the scalars of an operation's info dict, the part worth storing.

    Args:
        info: An operation's diagnostics dict, or None.
    """
    kept = {}
    for key, value in (info or {}).items():
        if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
            value = value.item()
        if isinstance(value, _SCALARS):
            kept[str(key)] = value
    return kept


def operation_labels(operations):
    """Return one label per operation: its name, with `_2`, `_3` added to repeats.

    Args:
        operations: A recipe's operations, in order.
    """
    seen = Counter()
    labels = []
    for operation in operations:
        seen[operation.name] += 1
        count = seen[operation.name]
        labels.append(operation.name if count == 1 else f"{operation.name}_{count}")
    return labels


def build_segment(
    image,
    mask=None,
    occurrence_id=None,
    part=DEFAULT_PART,
    project_path=None,
    panel_sink=None,
    from_mask=None,
    from_part=None,
):
    """Build one Segment ready for a transform chain.

    With `from_mask`, the segment starts as the upstream part's, so the chain frames that
    part; it is relabelled to `part` afterwards, with `mask` reprojected into the result.

    Args:
        image: The occurrence's image, BGR.
        mask: `part`'s own mask in original coordinates, or None.
        occurrence_id: As in `Segment`.
        part: As in `Segment`.
        project_path: As in `Segment`.
        panel_sink: As in `Segment`.
        from_mask: The upstream part's mask in original coordinates.
        from_part: The upstream part's name.
    """
    if from_mask is None:
        return Segment(
            image,
            mask=mask,
            occurrence_id=occurrence_id,
            part=part,
            project_path=project_path,
            panel_sink=panel_sink,
        )

    return Segment(
        image,
        mask=from_mask,
        occurrence_id=occurrence_id,
        part=from_part or DEFAULT_PART,
        project_path=project_path,
        panel_sink=panel_sink,
    )


def iterate_segments(
    project_path,
    part=DEFAULT_PART,
    transforms=(),
    reference=False,
    subset=None,
    limit=None,
    occurrence_ids=None,
    require_mask=True,
    from_part=None,
    report=pipeline_visualization.NULL_REPORT,
    mask_rows=None,
    tally=None,
    progress=None,
):
    """Yield `(occurrence_id, Segment)` for every occurrence with a mask for `part`.

    Args:
        project_path: Project to read from.
        part: Part whose masks to load.
        transforms: Operations applied to each segment before it is yielded.
        reference: Take masks from the reference table instead of the canonical one.
        subset: Named subset to restrict to.
        limit: Cap on how many occurrences are considered.
        occurrence_ids: Explicit ids, overriding `subset` and `limit`.
        require_mask: False also yields occurrences with no mask, with `segment.mask` None.
        from_part: Frame each segment by this part's canonical mask, then swap in `part`'s
            own mask reprojected into that frame (see `build_segment`).
        report: Visualization report; sampled segments use it as their panel sink.
        mask_rows: `{occurrence_id: row}` already loaded by the caller, to avoid a second read.
        tally: `Tally` to count `no_input`, failures and reliability flags into. Counting
            `processed` is left to the caller.
        progress: Label for the progress lines; defaults to naming the part.
    """
    paths.require_project(project_path)
    tally = tally if tally is not None else Tally()

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset, limit=limit)
    occurrence_ids = [str(occurrence_id) for occurrence_id in occurrence_ids]

    if mask_rows is None:
        mask_rows = mask_records.mask_lookup(
            project_path, part=part, occurrence_ids=occurrence_ids, reference=reference
        )
    source_rows = {}
    if from_part is not None:
        source_rows = mask_records.mask_lookup(project_path, part=from_part, occurrence_ids=occurrence_ids)

    missing = Counter()
    report.begin(occurrence_ids)
    progress = Progress(len(occurrence_ids), progress or f"segments part '{part}'", tallies=[tally])
    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in occurrence_ids:
            row = mask_rows.get(occurrence_id)
            if row is None and require_mask:
                tally.no_input += 1
                report.done(occurrence_id)
                progress.step()
                continue
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise NoInput(NO_IMAGE)

                mask = None if row is None else mask_records.decode_mask(row)
                from_mask = None
                if from_part is not None:
                    source_row = source_rows.get(occurrence_id)
                    if source_row is None:
                        raise NoInput(no_mask(from_part))
                    from_mask = mask_records.decode_mask(source_row)

                segment = build_segment(
                    image,
                    mask=mask,
                    occurrence_id=occurrence_id,
                    part=part,
                    project_path=project_path,
                    panel_sink=report.sink(occurrence_id),
                    from_mask=from_mask,
                    from_part=from_part,
                )

                for operation in transforms:
                    segment, info = operation(segment)
                    tally.record_flags(info)

                if from_mask is not None:
                    segment = segment.for_part(part)
                    segment.mask = None if mask is None else segment.project_mask(mask)

            except NoInput as exc:
                tally.no_input += 1
                missing[str(exc)] += 1
                report.done(occurrence_id)
                progress.step()
                continue
            except Exception as exc:
                logger.warning("skipping %s: %s", occurrence_id, exc)
                tally.record_failure(occurrence_id, exc)
                report.failure(occurrence_id, exc)
                report.done(occurrence_id)
                progress.step()
                continue

            yield occurrence_id, segment
            report.done(occurrence_id)
            progress.step()

    progress.finish()
    log_no_input(missing, f"part '{part}'")
