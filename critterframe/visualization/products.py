"""Products: render_segments, one image file per occurrence-part for downstream use."""

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
    """Return an occurrence column's value as the leading piece of a filename.

    Anything but letters, digits, `.` and `-` becomes `_`, and runs of `_` collapse to
    one. A missing or empty value becomes `MISSING_LABEL`.

    Args:
        value: The column value.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return MISSING_LABEL
    return re.sub(r"[^\w.\-]+|_+", "_", str(value)).strip("_") or MISSING_LABEL


def with_alpha(segment):
    """Return a segment's image with its mask as an alpha channel.

    Args:
        segment: The segment; without a mask, its image is returned unchanged.
    """
    image = np.asarray(segment.image)
    if segment.mask is None:
        return image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    alpha = np.where(segment.mask, 255, 0).astype(np.uint8)
    return np.dstack([image[:, :, :3], alpha])


def render_segments(
    project_path,
    name,
    transforms=(),
    part=DEFAULT_PART,
    parts=None,
    subset=None,
    limit=None,
    occurrence_ids=None,
    reference=False,
    extension=DEFAULT_FORMAT,
    force=False,
    from_part=None,
    visualize=True,
    visualize_every=None,
    name_by=None,
    transparent=True,
):
    """Write one image file per occurrence-part, after a chain of transforms.

    Occurrence-parts whose file exists are skipped unless `force`.

    Args:
        project_path: Project to read from and write into.
        name: The render's name, which starts its folder's name.
        transforms: Ordered transforms applied before writing; empty writes the originals.
        part: The part to render.
        parts: Several parts to render; each filename is then qualified with its part.
        subset: Named subset to render.
        limit: Cap on occurrences.
        occurrence_ids: Render exactly these, ignoring `subset` and `limit`.
        reference: Render from the reference masks.
        extension: Image format.
        force: Render again where the file exists.
        from_part: Frame each render by this upstream part's canonical mask.
        visualize: True, an int, or ids: a pipeline grid of what was rendered.
        visualize_every: Also write a grid every N occurrence-parts.
        name_by: Occurrence column whose value leads each filename, e.g.
            `Aeshna_cyanea__<occurrence_id>.png`. Such a render gets its own folder.
        transparent: Write the mask as an alpha channel. Has no effect, and logs so, for a
            format without alpha.

    Returns:
        `{part: summary}`, each a `drivers.Tally.summary` plus `directory`.
    """
    paths.require_project(project_path)

    target_parts = list(parts) if parts else [part]
    qualify = len(target_parts) > 1

    alpha = transparent and str(extension).lstrip(".").lower() in ALPHA_FORMATS
    if transparent and not alpha:
        logger.info("render '%s': '%s' files have no transparency -- writing opaque", name, extension)

    recipe = Recipe(
        "render",
        name,
        list(transforms),
        part=part,
        from_part=from_part,
        inputs={
            "masks": "reference" if reference else "canonical",
            "parts": sorted(target_parts),
            "format": extension,
            # Only when set, so a render made before this
            # existed keeps its folder.
            **({"name_by": name_by} if name_by is not None else {}),
            # Only when it applies, so an opaque render keeps
            # the folder it always had and no folder holds both.
            **({"transparent": True} if alpha else {}),
        },
    )
    directory = paths.products_dir(project_path, f"{name}_{recipe.hash}")
    directory.mkdir(parents=True, exist_ok=True)

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset, limit=limit)
    else:
        occurrence_ids = [str(occurrence_id) for occurrence_id in occurrence_ids]

    labels = {}
    if name_by is not None:
        occurrences = load_occurrences(project_path)
        if name_by not in occurrences.columns:
            raise KeyError(f"no '{name_by}' column to name files by (columns: {sorted(occurrences.columns)})")
        named = occurrences.set_index(ID_COL)[name_by]
        labels = {occurrence_id: file_label(named.get(occurrence_id)) for occurrence_id in occurrence_ids}

    logger.info(
        "render '%s': %d occurrence(s), part(s): %s -> %s",
        name,
        len(occurrence_ids),
        ", ".join(target_parts),
        directory,
    )

    report = pipeline_visualization.open_report(
        project_path,
        f"render__{name}",
        recipe.hash,
        visualize=visualize,
        visualize_every=visualize_every,
        identity=recipe.spec(),
    )

    results = {}
    for target_part in target_parts:
        tally = drivers.Tally(attempted=len(occurrence_ids))
        destinations = {
            occurrence_id: directory
            / product_filename(
                occurrence_id, target_part if qualify else None, extension, label=labels.get(occurrence_id)
            )
            for occurrence_id in occurrence_ids
        }
        pending = [
            occurrence_id for occurrence_id, dest in destinations.items() if force or not dest.exists()
        ]
        tally.skipped += len(occurrence_ids) - len(pending)

        for occurrence_id, segment in segment_iteration.iterate_segments(
            project_path,
            part=target_part,
            transforms=recipe.operations,
            reference=reference,
            occurrence_ids=pending,
            from_part=from_part,
            report=report,
            tally=tally,
            progress=f"render_segments '{name}' part '{target_part}'",
        ):
            try:
                rendered = with_alpha(segment) if alpha else segment.image
                if not cv2.imwrite(str(destinations[occurrence_id]), rendered):
                    raise ValueError(f"could not write {destinations[occurrence_id]}")
                tally.processed += 1
                segment.emit_panel(segment.image, "rendered")
            except Exception as exc:
                tally.record_failure(occurrence_id, exc)
                report.failure(occurrence_id, exc)
                logger.warning("render failed for %s part '%s': %s", occurrence_id, target_part, exc)

        logger.info(
            "render '%s' part '%s' complete: rendered=%d skipped=%d failed=%d",
            name,
            target_part,
            tally.processed,
            tally.skipped,
            tally.failed,
        )
        results[target_part] = tally.summary(directory=directory)

    report.close()
    return results
