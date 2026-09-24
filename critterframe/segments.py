"""
iterate_segments(): the per-occurrence loop most drivers walk, plus build_segment and how an operation's info is stored.

Open the image store once, build a Segment per occurrence (optionally framed by an upstream part's
mask), run a transform chain over it, hand it to the caller, and count what failed. Render,
validation and dataset-export drivers walk it directly; run_segments and run_metrics keep their own
loops (a multi-part fork; per-occurrence writes) but build with the same pieces.
"""

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
    """
    The part of an operation's `info` worth storing: scalars only.

    numpy scalars become Python ones; lists, dicts and arrays (a box, prompt
    points) are dropped, since a stored diagnostic is something to filter on.

    - `info` -- an operation's diagnostics dict, or None.
    """
    kept = {}
    for key, value in (info or {}).items():
        if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
            value = value.item()
        if isinstance(value, _SCALARS):
            kept[str(key)] = value
    return kept


def operation_labels(operations):
    """
    One label per operation, its name, with `_2`, `_3` added to repeats.

    What a stored diagnostic is keyed by, on a mask row and in a metric run
    alike, so `orient` means the same step on both sides.

    - `operations` -- a recipe's operations, in order.
    """
    seen = Counter()
    labels = []
    for operation in operations:
        seen[operation.name] += 1
        count = seen[operation.name]
        labels.append(operation.name if count == 1 else f"{operation.name}_{count}")
    return labels


def build_segment(image, mask=None, occurrence_id=None, part=DEFAULT_PART,
                  project_path=None, panel_sink=None, from_mask=None, from_part=None):
    """
    One Segment ready for a transform chain, framed by an upstream part's mask where asked.

    With `from_mask`, the segment starts as that part's segment -- so the chain
    crops and orients the shared upstream frame -- and is relabelled to `part`
    afterwards, with `mask` reprojected into the resulting frame. That is the
    frame a part carved out of another was trained and segmented in, so
    measuring or rendering it has to reproduce the same one rather than
    re-deriving a crop from the part's own, usually much smaller, mask.

    - `image` -- the occurrence's image, BGR.
    - `mask` -- `part`'s own mask in original coordinates, or None.
    - `occurrence_id`, `part`, `project_path`, `panel_sink` -- as `Segment`.
    - `from_mask` -- the upstream part's mask in original coordinates.
    - `from_part` -- that part's name, for the segment's label while the shared
      steps run.

    Returns a Segment.
    """
    if from_mask is None:
        return Segment(image, mask=mask, occurrence_id=occurrence_id, part=part,
                       project_path=project_path, panel_sink=panel_sink)

    return Segment(image, mask=from_mask, occurrence_id=occurrence_id,
                   part=from_part or DEFAULT_PART, project_path=project_path,
                   panel_sink=panel_sink)


def iterate_segments(project_path, part=DEFAULT_PART, transforms=(),
                     reference=False, subset=None, limit=None,
                     occurrence_ids=None, require_mask=True, from_part=None,
                     report=pipeline_visualization.NULL_REPORT, mask_rows=None,
                     tally=None, progress=None):
    """
    Yield (occurrence_id, Segment) for every occurrence with a mask for `part`.

    - `project_path` -- project to read from.
    - `part` -- part whose masks to load.
    - `transforms` -- operations applied to each segment before it's yielded.
    - `reference` -- take masks from the reference table rather than the
      canonical one. Usually True for training: the whole reason to train a
      model is that the automated masks weren't good enough, so training on
      them would teach the new model the old one's mistakes.
    - `subset` -- restrict to a named subset.
    - `limit` -- optional cap.
    - `occurrence_ids` -- explicit ids, overriding subset/limit selection.
    - `require_mask` -- False yields occurrences with no mask for `part`, with
      `segment.mask` left None, instead of skipping them. For training on
      whole images, where an unsegmented occurrence is still training data.
      A mask-dependent transform chain then fails per occurrence.
    - `from_part` -- build each segment from THIS part's CANONICAL mask
      instead of `part`'s own, run `transforms` against it, then swap in
      `part`'s own mask reprojected into the resulting frame (see
      `build_segment`). Always canonical regardless of `reference`: the
      upstream mask a hand-drawn correction started from was never itself
      written to the reference table, only the correction was.
    - `report` -- a visualization Report; sampled segments carry it as their
      panel_sink, so transform panels reach its grid, and each item is marked
      done once the consumer has handled it.
    - `mask_rows` -- `{occurrence_id: row}` already loaded by the caller, to
      avoid a second `mask_lookup` where the caller needed them anyway.
    - `tally` -- a `Tally` to count into: occurrences with nothing to work
      from as `no_input`, raised exceptions as failures, and every
      operation's reliability flags. Counting `processed` is the consumer's,
      since only it knows whether it kept the segment.
    - `progress` -- the label for periodic progress lines (see `Progress`), e.g.
      `"render_segments part 'organism'"`. Defaults to naming the part.
    """
    paths.require_project(project_path)
    tally = tally if tally is not None else Tally()

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                     limit=limit)
    occurrence_ids = [str(occurrence_id) for occurrence_id in occurrence_ids]

    if mask_rows is None:
        mask_rows = mask_records.mask_lookup(project_path, part=part,
                                             occurrence_ids=occurrence_ids,
                                             reference=reference)
    source_rows = {}
    if from_part is not None:
        source_rows = mask_records.mask_lookup(project_path, part=from_part,
                                               occurrence_ids=occurrence_ids)

    missing = Counter()
    report.begin(occurrence_ids)
    progress = Progress(len(occurrence_ids), progress or f"segments part '{part}'",
                        tallies=[tally])
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
                    image, mask=mask, occurrence_id=occurrence_id, part=part,
                    project_path=project_path,
                    panel_sink=report.sink(occurrence_id),
                    from_mask=from_mask, from_part=from_part)

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
