"""
Assets for downstream use: render_segments, one file per occurrence-part.

The opposite contract to pipeline: outputs, not diagnostics. Loose files rather
than the image store, because these are for figures and for R.

A render derives nothing and records nothing -- no mask, no metric, no run row.
It hashes its transform chain only so the folder name identifies what is in it
and a rerun is a no-op.
"""

import logging

import cv2

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment
from ..records import masks as mask_records
from ..storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "png"


# The filename contract lives with the rest of the project layout, in
# project.paths -- re-exported here because this is the module that writes the
# files, and a caller reasoning about a render's output looks here first.
product_filename = paths.product_filename


def render_segments(project_path, name, transforms=(), part=DEFAULT_PART,
                    parts=None, subset=None, limit=None, occurrence_ids=None,
                    reference=False, extension=DEFAULT_FORMAT, force=False):
    """
    Render each occurrence-part's segment through a chain of transforms and write
    one image file per occurrence-part.

    project_path -- project to read from and write into.
    name         -- the render's name, and its folder's, e.g. "oriented_bodies".
    transforms   -- ordered Transform operations applied before writing. The
                    interesting part: what makes a plate of comparable specimens
                    out of a pile of snapshots is usually remove_background() +
                    crop_to_mask() + orient(). An empty chain writes the
                    originals back out.
    part / parts -- the part to render, or several. Several qualifies every
                    filename with the part name.
    subset       -- name of a subset to render, or None for every occurrence.
    limit        -- optional cap, for checking a chain before committing to
                    10,000 files.
    occurrence_ids -- render exactly these, ignoring subset/limit.
    reference    -- render from the reference masks instead of the canonical.
    extension    -- image format by extension. PNG by default, since a render
                    usually goes into a figure and JPEG rings along the specimen
                    boundary.
    force        -- re-render occurrence-parts whose file exists. Normally
                    skipped, which makes an interrupted render resumable; output
                    is deterministic for a given hash, so skipping can't leave a
                    stale file.

    Returns a summary dict (rendered, skipped, failed, directory).
    """
    paths.require_project(project_path)

    target_parts = list(parts) if parts else [part]
    qualify = len(target_parts) > 1

    recipe = Recipe("render", name, list(transforms), part=part,
                    inputs={"masks": "reference" if reference else "canonical",
                            "parts": sorted(target_parts),
                            "format": extension})
    directory = paths.products_dir(project_path, f"{name}_{recipe.hash}")
    directory.mkdir(parents=True, exist_ok=True)

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                     limit=limit)
    else:
        occurrence_ids = [str(occurrence_id) for occurrence_id in occurrence_ids]

    logger.info("render '%s': %d occurrence(s), part(s): %s -> %s",
                name, len(occurrence_ids), ", ".join(target_parts), directory)

    rendered = 0
    skipped = 0
    failed = 0

    with ImageStore(project_path, readonly=True) as images:
        for target_part in target_parts:
            mask_rows = mask_records.mask_lookup(project_path, part=target_part,
                                                 occurrence_ids=occurrence_ids,
                                                 reference=reference)
            for occurrence_id in occurrence_ids:
                dest = directory / product_filename(
                    occurrence_id, target_part if qualify else None, extension)
                if dest.exists() and not force:
                    skipped += 1
                    continue

                mask_row = mask_rows.get(occurrence_id)
                if mask_row is None:
                    # Not a failure: a project legitimately has parts that only
                    # some occurrences carry, and a render of "every wing" over
                    # a project where half the specimens have none should write
                    # the wings it has and say how many it didn't.
                    skipped += 1
                    continue

                try:
                    image = images.get(occurrence_id)
                    if image is None:
                        raise ValueError("no image in the image store")

                    state = Segment(image,
                                    mask=mask_records.decode_mask(mask_row),
                                    occurrence_id=occurrence_id,
                                    part=target_part,
                                    project_path=project_path)
                    for operation in recipe.operations:
                        state, _info = operation(state)

                    if not cv2.imwrite(str(dest), state.image):
                        raise ValueError(f"could not write {dest}")
                    rendered += 1

                except Exception as exc:
                    failed += 1
                    logger.warning("render failed for %s part '%s': %s",
                                   occurrence_id, target_part, exc)

    logger.info("render '%s' complete: rendered=%d skipped=%d failed=%d",
                name, rendered, skipped, failed)
    return {"rendered": rendered, "skipped": skipped, "failed": failed,
            "directory": directory}
