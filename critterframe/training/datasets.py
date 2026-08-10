"""
Build training datasets out of a project.

A project accumulates exactly what a segmenter or an encoder needs to be
trained on -- images, masks, parts, labels -- so training data isn't something
to assemble separately, it's a view over what's already stored. That's the path
from "CritterFrame segments whole organisms with a general model" to
"CritterFrame segments THIS project's abdomens with a model trained here":
segment organisms, correct a few dozen by hand into reference masks, build a
dataset from those, train, and plug the result back in as segment(my_model).

Two shapes, because training code comes in two shapes:

  iterate_segments() -- yields Segment objects, for a torch Dataset or any
                        in-memory training loop. Nothing is written.
  write_dataset()    -- writes image/mask PNG pairs and a manifest, for
                        training code that wants files on disk.

Both take the same `transforms`, and using the same chain here as the model
will get at inference time is the point: a model trained on background-removed,
oriented segments needs those same operations in front of it when it runs, and
sharing one operation list is what keeps the two from drifting apart.
"""

import logging
import os

import cv2
import pandas as pd

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Segment
from ..records import masks as mask_records
from ..export import metrics_wide
from ..storage.imagestore import ImageStore

logger = logging.getLogger(__name__)


def iterate_segments(project_path, part=DEFAULT_PART, transforms=(),
                     reference=False, subset=None, limit=None,
                     occurrence_ids=None):
    """
    Yield (occurrence_id, Segment) for every occurrence with a mask for `part`.

    project_path   -- project to read from.
    part           -- part whose masks to load.
    transforms     -- operations applied to each segment before it's yielded.
    reference      -- take masks from the reference table rather than the
                      canonical one. Usually True for training: the whole
                      reason to train a model is that the automated masks
                      weren't good enough, so training on them would teach the
                      new model to reproduce the old one's mistakes.
    subset         -- restrict to a named subset.
    limit          -- optional cap.
    occurrence_ids -- explicit ids, overriding subset/limit selection.
    """
    paths.require_project(project_path)

    if occurrence_ids is None:
        occurrence_ids = subset_selection.select_ids(project_path, subset=subset,
                                                     limit=limit)
    occurrence_ids = [str(i) for i in occurrence_ids]

    mask_rows = mask_records.mask_lookup(project_path, part=part,
                                         occurrence_ids=occurrence_ids,
                                         reference=reference)

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id in occurrence_ids:
            row = mask_rows.get(occurrence_id)
            if row is None:
                continue
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                segment = Segment(image, mask=mask_records.decode_mask(row),
                                  occurrence_id=occurrence_id, part=part,
                                  project_path=project_path)
                for operation in transforms:
                    segment, _info = operation(segment)

                yield occurrence_id, segment

            except Exception as exc:
                logger.warning("skipping %s: %s", occurrence_id, exc)


def write_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                  reference=False, subset=None, limit=None, label_columns=None,
                  label_runs=None):
    """
    Write an image/mask pair per occurrence plus a manifest CSV.

    Layout:
        output_dir/
            images/<occurrence_id>.png
            masks/<occurrence_id>.png
            manifest.csv

    Masks are written as 0/255 PNGs -- lossless, so a mask boundary survives
    the round trip exactly. Images are PNG too: a model trained on re-JPEGed
    images learns the compression artifacts along with the organism.

    output_dir    -- directory to write into; created if missing.
    label_columns -- occurrence-table columns to carry into the manifest
                    (species, sex, collection...), for training something
                    supervised on them.
    label_runs    -- metric run names whose values to carry into the manifest,
                    so a stored metric can be a training target or a
                    stratification key.

    Everything else is as iterate_segments().

    Returns the manifest DataFrame.
    """
    image_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rows = []
    for occurrence_id, segment in iterate_segments(
            project_path, part=part, transforms=transforms, reference=reference,
            subset=subset, limit=limit):
        image_path = os.path.join(image_dir, f"{occurrence_id}.png")
        mask_path = os.path.join(mask_dir, f"{occurrence_id}.png")

        cv2.imwrite(image_path, segment.image)
        cv2.imwrite(mask_path, segment.mask.astype("uint8") * 255)

        rows.append({
            "occurrence_id": occurrence_id,
            "part": part,
            "image_path": os.path.relpath(image_path, output_dir),
            "mask_path": os.path.relpath(mask_path, output_dir),
            "height": segment.shape[0],
            "width": segment.shape[1],
            "mask_area": int(segment.mask.sum()),
        })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        logger.warning("no segments written -- does part '%s' have %s masks?",
                       part, "reference" if reference else "canonical")
        return manifest

    manifest = _attach_labels(project_path, manifest, label_columns, label_runs,
                              part)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    logger.info("wrote %d training pairs -> %s", len(manifest), output_dir)
    return manifest


def _attach_labels(project_path, manifest, label_columns, label_runs, part):
    """Join occurrence metadata and stored metric values onto a manifest."""
    if label_columns:
        occurrences = subset_selection.select_occurrences(
            project_path, columns=list(label_columns))
        manifest = manifest.merge(occurrences, on="occurrence_id", how="left")

    if label_runs:
        values = metrics_wide(project_path, run_names=list(label_runs),
                              parts=[part])
        if not values.empty:
            manifest = manifest.merge(values, on="occurrence_id", how="left")

    return manifest
