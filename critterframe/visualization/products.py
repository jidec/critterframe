"""
Assets for downstream use: render_segments, one file per occurrence-part.

The opposite contract to pipeline: outputs, not diagnostics. Loose files rather than the image
store, because these are for figures and for R.

A render derives nothing and records nothing -- no mask, no metric, no run row. It hashes its
transform chain only so the folder name identifies what is in it and a rerun is a no-op.
"""

import logging
import re

import cv2
import numpy as np
import pandas as pd

from .. import drivers, segments as segment_iteration
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe
from ..records.occurrences import ID_COL, load_occurrences
from . import pipeline as pipeline_visualization

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "png"

# What a filename says where the `name_by` column has no value.
MISSING_LABEL = "unknown"

# Formats that can carry the mask as an alpha channel.
ALPHA_FORMATS = {"png", "webp", "tif", "tiff"}


# The filename contract lives with the rest of the project layout, in
# project.paths -- re-exported here because this is the module that writes the
# files, and a caller reasoning about a render's output looks here first.
product_filename = paths.product_filename


def file_label(value):
    """
    An occurrence column's value as the leading piece of a filename.

    Anything but letters, digits, `.` and `-` becomes `_`, and runs of `_`
    collapse to one, so a double underscore only ever separates pieces.

    Returns the label; `MISSING_LABEL` for a missing or empty value.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return MISSING_LABEL
    return re.sub(r"[^\w.\-]+|_+", "_", str(value)).strip("_") or MISSING_LABEL


def with_alpha(segment):
    """
    A segment's image with its mask as an alpha channel: opaque inside, transparent outside.

    Returns a uint8 BGRA array; the image unchanged where the segment has no mask.
    """
    image = np.asarray(segment.image)
    if segment.mask is None:
        return image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    alpha = np.where(segment.mask, 255, 0).astype(np.uint8)
    return np.dstack([image[:, :, :3], alpha])


def render_segments(project_path, name, transforms=(), part=DEFAULT_PART,
                    parts=None, subset=None, limit=None, occurrence_ids=None,
                    reference=False, extension=DEFAULT_FORMAT, force=False,
                    from_part=None, visualize=True, visualize_every=None,
                    name_by=None, transparent=True):
    """
    Render each occurrence-part's segment through a chain of transforms and write
    one image file per occurrence-part.

    - `project_path` -- project to read from and write into.
    - `name` -- the render's name, and its folder's, e.g.
      `"oriented_bodies"`.
    - `transforms` -- ordered Transform operations applied before writing.
      The interesting part: what makes a plate of comparable specimens out
      of a pile of snapshots is usually `remove_background()` +
      `crop_to_mask()` + `orient()`. An empty chain writes the originals
      back out.
    - `part` / `parts` -- the part to render, or several. Several qualifies
      every filename with the part name.
    - `subset` -- name of a subset to render, or None for every occurrence.
    - `limit` -- optional cap, for checking a chain before committing to
      10,000 files.
    - `occurrence_ids` -- render exactly these, ignoring subset/limit.
    - `reference` -- render from the reference masks instead of the
      canonical.
    - `extension` -- image format by extension. PNG by default, since a
      render usually goes into a figure and JPEG rings along the specimen
      boundary.
    - `force` -- re-render occurrence-parts whose file exists. Normally
      skipped, which makes an interrupted render resumable; output is
      deterministic for a given hash, so skipping can't leave a stale file.
    - `from_part` -- frame each render by an upstream part's CANONICAL mask
      instead of the rendered part's own, so a part carved out of another is
      rendered in the shared crop it was segmented in (see
      `segments.iterate_segments`).
    - `visualize` -- True (default), an int, or ids: a pipeline grid of what
      was rendered, with every failure listed in its sidecar. False writes
      nothing.
    - `visualize_every` -- also write a grid every N occurrence-parts.
    - `name_by` -- an occurrence column, e.g. `"species"`, whose value leads
      each filename: `Aeshna_cyanea__<occurrence_id>.png`. A missing value
      reads `unknown`. A render named this way gets a folder of its own.
    - `transparent` -- write the mask as an alpha channel, so everything
      outside the segment is transparent and the file sits on any
      background. Ignored, with a log line, for a format that has no alpha
      (JPEG). False writes the opaque image, in the folder such a render
      always had.

    Returns {part: summary}, each as `drivers.Tally.summary` plus
    `directory` -- the same shape `run_segments` and `run_metrics` return, one
    entry even when only `part` was given.
    """
    paths.require_project(project_path)

    target_parts = list(parts) if parts else [part]
    qualify = len(target_parts) > 1

    alpha = transparent and str(extension).lstrip(".").lower() in ALPHA_FORMATS
    if transparent and not alpha:
        logger.info("render '%s': '%s' files have no transparency -- writing opaque",
                    name, extension)

    recipe = Recipe("render", name, list(transforms), part=part,
                    from_part=from_part,
                    inputs={"masks": "reference" if reference else "canonical",
                            "parts": sorted(target_parts),
                            "format": extension,
                            # Only when set, so a render made before this
                            # existed keeps its folder.
                            **({"name_by": name_by} if name_by is not None else {}),
                            # Only when it applies, so an opaque render keeps
                            # the folder it always had and no folder holds both.
                            **({"transparent": True} if alpha else {})})
    directory = paths.products_dir(project_path, f"{name}_{recipe.hash}")
    directory.mkdir(parents=True, exist_ok=True)

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                     limit=limit)
    else:
        occurrence_ids = [str(occurrence_id) for occurrence_id in occurrence_ids]

    labels = {}
    if name_by is not None:
        occurrences = load_occurrences(project_path)
        if name_by not in occurrences.columns:
            raise KeyError(f"no '{name_by}' column to name files by "
                           f"(columns: {sorted(occurrences.columns)})")
        named = occurrences.set_index(ID_COL)[name_by]
        labels = {occurrence_id: file_label(named.get(occurrence_id))
                  for occurrence_id in occurrence_ids}

    logger.info("render '%s': %d occurrence(s), part(s): %s -> %s",
                name, len(occurrence_ids), ", ".join(target_parts), directory)

    report = pipeline_visualization.open_report(
        project_path, f"render__{name}", recipe.hash, visualize=visualize,
        visualize_every=visualize_every, identity=recipe.spec())

    results = {}
    for target_part in target_parts:
        tally = drivers.Tally(attempted=len(occurrence_ids))
        destinations = {
            occurrence_id: directory / product_filename(
                occurrence_id, target_part if qualify else None, extension,
                label=labels.get(occurrence_id))
            for occurrence_id in occurrence_ids
        }
        pending = [occurrence_id for occurrence_id, dest in destinations.items()
                   if force or not dest.exists()]
        tally.skipped += len(occurrence_ids) - len(pending)

        for occurrence_id, segment in segment_iteration.iterate_segments(
                project_path, part=target_part, transforms=recipe.operations,
                reference=reference, occurrence_ids=pending, from_part=from_part,
                report=report, tally=tally,
                progress=f"render_segments '{name}' part '{target_part}'"):
            try:
                rendered = with_alpha(segment) if alpha else segment.image
                if not cv2.imwrite(str(destinations[occurrence_id]), rendered):
                    raise ValueError(f"could not write {destinations[occurrence_id]}")
                tally.processed += 1
                segment.emit_panel(segment.image, "rendered")
            except Exception as exc:
                tally.record_failure(occurrence_id, exc)
                report.failure(occurrence_id, exc)
                logger.warning("render failed for %s part '%s': %s",
                               occurrence_id, target_part, exc)

        logger.info("render '%s' part '%s' complete: rendered=%d skipped=%d "
                    "failed=%d", name, target_part, tally.processed,
                    tally.skipped, tally.failed)
        results[target_part] = tally.summary(directory=directory)

    report.close()
    return results
