"""
Products: visual assets deliberately materialized, one file per occurrence-part.

A render takes reproducible project state -- an image, its canonical mask, a
chain of transforms -- and writes what that looks like as an ordinary image
file. Nothing else happens: no mask is written, no metric is stored, no run is
recorded. A picture of the data is not a measurement of it, and treating it as
one would put rows in the metric log that no analysis should ever read.

    render_segments(
        project_path,
        name="oriented_bodies",
        transforms=[remove_background(), crop_to_mask(pad=10), orient()],
    )

    visualizations/products/oriented_bodies_<hash>/
        specimen0001.png
        specimen0002.png
        ...

The contract is one file per occurrence-part, named for the occurrence. That's
what makes a product folder joinable to an exported trait table by anything that
can list a directory -- R reading filenames into a column, a figure script
looking up one specimen, a person sorting a folder by name. A render that
covered several parts qualifies each filename with the part
(`specimen0001__forewing_left.png`), so the mapping stays one row per file
either way.

Loose files rather than the image store, deliberately. The store exists to hold
original analysis images byte-exactly, keyed by occurrence id, for Python to
read; a render is derived, regenerable, and usually wanted by something that
isn't Python. Putting it in LMDB would make it harder to use for the exact
purpose it was made for.

The transform chain is hashed the same way any recipe is, and the hash names the
folder, so two crops of the same specimens coexist instead of overwriting each
other, and re-running an unchanged render is a no-op.
"""

import logging

import cv2

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Recipe, Segment
from ..records import masks as mask_records
from ..storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "png"


def product_filename(occurrence_id, part=None, extension=DEFAULT_FORMAT):
    """
    The filename one rendered occurrence-part gets.

    part=None writes `<occurrence_id>.<ext>`, for a render covering a single
    part; naming it `<occurrence_id>__<part>.<ext>` otherwise. Both forms put
    the occurrence id first and separate the part with a double underscore, so
    splitting a filename back into ids is one operation in any language.
    """
    stem = str(occurrence_id) if part is None else f"{occurrence_id}__{part}"
    return f"{stem}.{extension.lstrip('.')}"


def render_segments(project_path, name, transforms=(), part=DEFAULT_PART,
                    parts=None, subset=None, limit=None, occurrence_ids=None,
                    reference=False, extension=DEFAULT_FORMAT, force=False):
    """
    Render each occurrence-part's segment through a chain of transforms and
    write one image file per occurrence-part.

    project_path -- project to read from and write into.
    name         -- the render's name, and its folder's, e.g. "oriented_bodies".
    transforms   -- ordered Transform operations applied to each segment before
                    it's written. The interesting part of a render: what makes
                    a plate of comparable specimens out of a pile of snapshots
                    is usually remove_background() + crop_to_mask() + orient().
                    An empty chain writes the original images back out, which is
                    valid and rarely what anyone wants.
    part / parts -- the part to render, or several. Rendering several parts
                    qualifies every filename with the part name (see
                    product_filename), including for parts that would otherwise
                    have been the default.
    subset       -- name of a subset to render, or None for every occurrence.
    limit        -- optional cap, for checking a chain looks right before
                    committing to 10,000 files.
    occurrence_ids -- render exactly these, ignoring subset/limit selection.
    reference    -- render from the reference masks instead of the canonical
                    ones.
    extension    -- image format, by file extension. PNG by default: a render is
                    usually going into a figure, where a lossless file with an
                    exact edge beats a smaller one with JPEG ringing along the
                    specimen boundary.
    force        -- re-render occurrence-parts whose file already exists.
                    Normally they're skipped, which makes an interrupted render
                    resumable; the output is deterministic for a given hash, so
                    skipping can't leave a stale file behind.

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
