"""
Build training datasets out of a project.

A project accumulates exactly what a segmenter or an encoder needs to be
trained on -- images, masks, parts, labels -- so training data isn't something
to assemble separately, it's a view over what's already stored. That's the path
from "CritterFrame segments whole organisms with a general model" to
"CritterFrame segments THIS project's abdomens with a model trained here":
segment organisms, correct a few dozen by hand into reference masks, build a
dataset from those, train, and plug the result back in as segment(my_model).

Three shapes, because training code comes in three shapes:

  iterate_segments()      -- yields Segment objects, for a torch Dataset or any
                             in-memory training loop. Nothing is written.
  export_training_data()  -- materializes a whole dataset: images (and masks)
                             laid out by split and optionally by class, plus a
                             manifest and a record of what was exported.
  write_dataset()         -- one flat directory of image/mask pairs and a
                             manifest; export_training_data() with no splits.

All three take the same `transforms`, and using the same chain here as the
model will get at inference time is the point: a model trained on
background-removed, oriented segments needs those same operations in front of
it when it runs, and sharing one operation list is what keeps the two from
drifting apart.

WHAT IS EXPORTED IS NOT WHAT DECIDED THE SPLIT. Nothing here chooses
proportions, stratification, or grouping -- it takes splits already decided
(training.splits.split_ids, or named subsets) and writes them out. Which
occurrences answer which question is a scientific decision that wants to be
made once, reviewed, and reused across every dataset built from the project;
folding it into the exporter would mean re-deciding it every time anything is
materialized, and two datasets from one project could then disagree about what
"test" means.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, Segment, hash_spec
from ..records import masks as mask_records
from ..records.occurrences import ID_COL, ids_digest
from ..export import metrics_wide
from ..storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.csv"

# What was exported, beside what was exported. Not a manifest (that is one row
# per file) and not a run record (nothing was derived): a dataset is an
# arrangement of existing data, and this is the note that says which
# arrangement, so a checkpoint trained from it can point at something more
# durable than a folder name (see records.models.register_model).
DATASET_FILE = "dataset.json"

IMAGE_DIR = "images"
MASK_DIR = "masks"

# Name for the whole export in the dataset record when no splits were asked
# for -- the record always describes at least one group, so anything reading it
# sees one shape.
UNSPLIT = "all"

# Characters allowed in a class folder name. A class comes from occurrence
# metadata and is written by whoever recorded it: "Anax junius" is fine, but
# "Aeshnidae/Anax" would silently become a nested directory and an ImageFolder
# loader would then report a class nobody has.
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def iterate_segments(project_path, part=DEFAULT_PART, transforms=(),
                     reference=False, subset=None, limit=None,
                     occurrence_ids=None, require_mask=True):
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
    require_mask   -- False yields occurrences that have no mask for `part`,
                      with segment.mask left None, instead of skipping them.
                      For training on whole images -- a classifier over
                      photographs the project never segmented -- where an
                      unsegmented occurrence is still training data. Any
                      transform that needs a mask then fails per occurrence and
                      is logged, so don't pass a mask-dependent chain with this.
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
            if row is None and require_mask:
                continue
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                mask = None if row is None else mask_records.decode_mask(row)
                segment = Segment(image, mask=mask,
                                  occurrence_id=occurrence_id, part=part,
                                  project_path=project_path)
                for operation in transforms:
                    segment, _info = operation(segment)

                yield occurrence_id, segment

            except Exception as exc:
                logger.warning("skipping %s: %s", occurrence_id, exc)


def export_training_data(project_path, output_dir, splits=None, part=DEFAULT_PART,
                         transforms=(), reference=False, masks=False,
                         class_by=None, metadata=None, metrics=None,
                         require_mask=True, subset=None, limit=None):
    """
    Materialize project data as a directory a trainer can read.

    Layout, with the split level present only when `splits` is given and the
    class level only when `class_by` is:

        output_dir/
            manifest.csv
            dataset.json
            train/<class>/<occurrence_id>.png     (class_by given)
            train/images/<occurrence_id>.png      (no class_by)
            train/masks/<occurrence_id>.png       (masks=True)
            val/...

    project_path -- project to export from.
    output_dir   -- directory to write into; created if missing. Existing files
                    are left alone unless a new one has the same name, so
                    export twice into one directory only if you mean to.
    splits       -- {split name: subset name} or {split name: occurrence ids},
                    mixed freely. None exports everything as one flat dataset.
                    THIS FUNCTION DOES NOT DECIDE THE SPLIT: pass what
                    training.splits.split_ids() returned, or the names of
                    subsets the project has frozen. An occurrence appearing in
                    two splits raises, because that is the leakage every other
                    guarantee here exists to prevent, and a written dataset is
                    exactly where it stops being visible.
    part         -- part to export, "organism" by default.
    transforms   -- operations applied to each segment before it is written.
                    Use the SAME chain the trained model will get at inference
                    time (see the module docstring).
    reference    -- read masks from the reference table rather than the
                    canonical one. Usually True when training a segmenter.
    masks        -- also write each mask as a 0/255 PNG, from whichever table
                    `reference` names. What segmentation training needs and
                    what a classifier has no use for, hence off by default.
    class_by     -- occurrence column to organize images into class folders by,
                    e.g. "species", producing an ImageFolder-shaped tree. An
                    occurrence with no value in that column is left out and the
                    count is logged: the class IS the training target here, so
                    an image without one has nothing to teach -- unlike a split,
                    where an unlabelled image is still data.
    metadata     -- occurrence columns to carry into the manifest.
    metrics      -- metric run names whose values to carry into the manifest,
                    so a stored trait or QC score can be a target or a filter
                    downstream.
    require_mask -- False exports occurrences that have no mask for `part`, for
                    training on whole images. See iterate_segments().
    subset, limit -- narrow an unsplit export. Rejected alongside `splits`,
                    which already says which occurrences are wanted.

    Returns the manifest DataFrame: one row per written image, carrying
    occurrence_id, part, the split and class where there are any, the written
    paths, and whatever metadata/metrics were asked for.
    """
    paths.require_project(project_path)

    if splits is not None and (subset is not None or limit is not None):
        raise ValueError(
            "export_training_data takes splits= or subset=/limit=, not both -- "
            "splits already says which occurrences to export"
        )

    selections = _resolve_splits(project_path, splits, subset, limit)
    class_values = _class_values(project_path, class_by)
    folders = {}

    rows = []
    unclassed = 0
    unwritten = 0

    for split_name, occurrence_ids in selections:
        for occurrence_id, segment in iterate_segments(
                project_path, part=part, transforms=transforms,
                reference=reference, occurrence_ids=occurrence_ids,
                require_mask=require_mask):

            class_value = None
            if class_by is not None:
                class_value = class_values.get(occurrence_id)
                if class_value is None:
                    unclassed += 1
                    continue

            leaf = _class_folder(folders, class_value) if class_by else IMAGE_DIR
            image_path = os.path.join(_directory(output_dir, split_name, leaf),
                                      f"{occurrence_id}.png")
            if not _write_image(image_path, segment.image):
                unwritten += 1
                continue

            row = {"occurrence_id": occurrence_id, "part": part}
            if splits is not None:
                row["split"] = split_name
            if class_by is not None:
                row["class"] = class_value
            row["image_path"] = _relative(image_path, output_dir)

            if masks:
                mask_path = None
                if segment.mask is not None:
                    mask_path = os.path.join(
                        _directory(output_dir, split_name, MASK_DIR),
                        f"{occurrence_id}.png")
                    # 0/255 PNG: lossless, so a mask boundary survives the
                    # round trip exactly, and readable by anything.
                    _write_image(mask_path, segment.mask.astype("uint8") * 255)
                    mask_path = _relative(mask_path, output_dir)
                row["mask_path"] = mask_path

            row["height"] = segment.shape[0]
            row["width"] = segment.shape[1]
            row["mask_area"] = (None if segment.mask is None
                                else int(segment.mask.sum()))
            rows.append(row)

    if unclassed:
        logger.info("left out %d occurrence(s) with no value in '%s'",
                    unclassed, class_by)
    if unwritten:
        logger.warning("%d image(s) could not be written -- check the "
                       "occurrence ids for characters the filesystem rejects",
                       unwritten)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        logger.warning("nothing exported -- does part '%s' have %s masks?",
                       part, "reference" if reference else "canonical")
        return manifest

    manifest = _attach_labels(project_path, manifest, metadata, metrics, part)
    manifest.to_csv(os.path.join(output_dir, MANIFEST_FILE), index=False)
    _write_dataset_record(output_dir, manifest, splits, part, transforms,
                          reference, masks, class_by)

    logger.info("exported %d image(s) -> %s", len(manifest), output_dir)
    return manifest


def write_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                  reference=False, subset=None, limit=None, label_columns=None,
                  label_runs=None):
    """
    Write an image/mask pair per occurrence plus a manifest CSV, all in one
    flat directory:

        output_dir/
            images/<occurrence_id>.png
            masks/<occurrence_id>.png
            manifest.csv

    export_training_data() with no splits and no class folders, kept under its
    own name because "give me every segment as files" is what training code
    that does its own splitting wants, and because it is what the BioEncoder
    extension is written against.

    label_columns -- occurrence columns to carry into the manifest
                     (export_training_data's `metadata`).
    label_runs    -- metric run names to carry into the manifest
                     (export_training_data's `metrics`).

    Everything else is as iterate_segments(). Returns the manifest DataFrame.
    """
    return export_training_data(
        project_path, output_dir, part=part, transforms=transforms,
        reference=reference, masks=True, metadata=label_columns,
        metrics=label_runs, subset=subset, limit=limit)


def _resolve_splits(project_path, splits, subset, limit):
    """
    Turn the `splits` argument into an ordered [(split name, ids)], checking
    that no occurrence is in two of them.

    A split value is a subset name or a list of ids, because both are how a
    project legitimately holds a selection: split_ids() hands back ids, while a
    selection worth keeping gets frozen as a subset. Resolving both here means
    the exporter never has to know which one a caller used.
    """
    if splits is None:
        return [(None, subset_selection.select_ids(project_path, subset=subset,
                                                   limit=limit))]

    resolved = []
    for name, selection in splits.items():
        if isinstance(selection, str):
            ids = subset_selection.select_ids(project_path, subset=selection)
        else:
            ids = [str(occurrence_id) for occurrence_id in selection]
        resolved.append((name, ids))

    seen = {}
    for name, ids in resolved:
        for occurrence_id in ids:
            if occurrence_id in seen and seen[occurrence_id] != name:
                raise ValueError(
                    f"occurrence {occurrence_id} is in both "
                    f"'{seen[occurrence_id]}' and '{name}' -- an occurrence in "
                    "two splits is training data leaking into evaluation, and a "
                    "written dataset is where that stops being visible"
                )
            seen[occurrence_id] = name

    logger.info("exporting %d split(s): %s", len(resolved),
                ", ".join(f"{name}={len(ids)}" for name, ids in resolved))
    return resolved


def _class_values(project_path, class_by):
    """
    {occurrence_id: class value} for the class column, or {} when there is no
    class column. Missing and blank values are simply absent from the mapping,
    which is what makes "has no class" one check at the call site.
    """
    if class_by is None:
        return {}

    occurrences = subset_selection.select_occurrences(project_path,
                                                      columns=[class_by])
    if class_by not in occurrences.columns:
        raise KeyError(
            f"occurrence table has no column '{class_by}' to make classes from "
            f"(columns: {sorted(occurrences.columns)})"
        )

    values = {}
    for occurrence_id, value in zip(occurrences[ID_COL], occurrences[class_by]):
        if pd.isna(value) or str(value).strip() == "":
            continue
        values[occurrence_id] = str(value)
    return values


def _class_folder(folders, class_value):
    """
    The folder name for a class, remembering the mapping so two classes cannot
    quietly share a folder.

    The collision check is why this is not a bare sanitizer: "Anax junius" and
    "Anax/junius" both flatten to Anax_junius, and merging two classes into one
    folder would train a model on a taxonomy nobody wrote, invisibly.
    """
    if class_value in folders:
        return folders[class_value]

    folder = _UNSAFE_IN_NAME.sub("_", class_value).strip("_") or "unnamed"
    clash = next((existing for existing, name in folders.items()
                  if name == folder), None)
    if clash is not None:
        raise ValueError(
            f"classes {clash!r} and {class_value!r} both become folder "
            f"'{folder}' -- rename one in the occurrence table, or export with "
            "a different class_by column"
        )

    folders[class_value] = folder
    return folder


def _directory(output_dir, split_name, leaf):
    """The directory one image belongs in, created on the way."""
    parts = [output_dir] + ([split_name] if split_name else []) + [leaf]
    directory = os.path.join(*parts)
    os.makedirs(directory, exist_ok=True)
    return directory


def _relative(path, output_dir):
    """
    A written file as the manifest should name it: relative to the dataset
    directory, with forward slashes.

    Relative so the whole directory can be moved or copied to wherever training
    happens, and posix-separated because that move is very often Windows to a
    Linux cluster, where a backslash is a legal filename character rather than
    a separator and the manifest would silently point at nothing.
    """
    return Path(os.path.relpath(path, output_dir)).as_posix()


def _write_image(path, image):
    """
    Write one PNG, reporting failure rather than raising.

    PNG throughout: a model trained on re-JPEGed images learns the compression
    artifacts along with the organism, and a re-encoded mask edge is simply
    wrong. cv2.imwrite returns False rather than raising on a path the
    filesystem will not take -- an occurrence id with a colon in it, on Windows
    -- and one such occurrence should not end an export of thousands.
    """
    if cv2.imwrite(path, image):
        return True
    logger.warning("could not write %s", path)
    return False


def _write_dataset_record(output_dir, manifest, splits, part, transforms,
                          reference, masks, class_by):
    """
    Write dataset.json: what this export IS, hashed.

    The splits are recorded as counts and id digests taken from the manifest --
    what was actually written, not what was asked for, since occurrences drop
    out for want of a mask or a class. `data_hash` covers the whole description
    except the timestamp, so two exports of the same occurrences through the
    same transforms hash alike, and a trained model's record can name the
    training data it saw rather than the directory it happened to sit in.
    """
    if splits is None:
        groups = {UNSPLIT: manifest["occurrence_id"].tolist()}
    else:
        groups = {name: frame["occurrence_id"].tolist()
                  for name, frame in manifest.groupby("split")}

    record = {
        "part": part,
        "reference": bool(reference),
        "masks": bool(masks),
        "class_by": class_by,
        "classes": (sorted(manifest["class"].unique().tolist())
                    if class_by is not None else None),
        "transforms": [operation.spec() for operation in transforms],
        "splits": {name: {"count": len(ids), "ids_hash": ids_digest(ids)}
                   for name, ids in sorted(groups.items())},
    }
    record["data_hash"] = hash_spec(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path = os.path.join(output_dir, DATASET_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


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
