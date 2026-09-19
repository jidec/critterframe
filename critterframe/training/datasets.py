"""
export_training_data(): images, masks, class folders, and a manifest.

Splitting decides which occurrences answer which question; exporting
materializes them. Training itself happens outside the package -- what comes
back is registered (records.models), and then segment() runs it like any other
model.

Export from REFERENCE masks where they exist: training on canonical masks
teaches a new model the old model's mistakes.
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from .. import segments
from ..project import paths, subsets as subset_selection
from ..recipes import DEFAULT_PART, hash_spec
from ..records.occurrences import ID_COL, ids_record
from ..export import metrics_wide
from ..storage.jsonfiles import write_json
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import segment_panel

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


# The per-occurrence loop lives in core `segments`, not here: a dataset export
# is only one of the drivers that walks it (see that module), and two colour
# metrics were already reaching into this module for it.
iterate_segments = segments.iterate_segments


def export_training_data(project_path, output_dir, splits=None, part=DEFAULT_PART,
                         transforms=(), reference=False, masks=False,
                         class_by=None, metadata=None, metrics=None,
                         require_mask=True, subset=None, limit=None,
                         from_part=None, visualize=True):
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

    - `project_path` -- project to export from.
    - `output_dir` -- directory to write into; created if missing. Existing
      files are left alone unless a new one has the same name.
    - `splits` -- `{split name: subset name}` or `{split name: occurrence
      ids}`, mixed freely. None exports everything as one flat dataset. This
      does NOT decide the split -- pass what `split_ids()` returned. An
      occurrence in two splits raises, since that is the leakage every
      other guarantee here exists to prevent.
    - `part` -- part to export, `"organism"` by default.
    - `transforms` -- operations applied to each segment before writing. Use
      the SAME chain the model will get at inference.
    - `reference` -- read masks from the reference table. Usually True when
      training a segmenter.
    - `masks` -- also write each mask as a 0/255 PNG.
    - `class_by` -- occurrence column to organize images into class folders
      by, e.g. `"species"`. An occurrence with no value is left out and
      logged: the class is the training target, so an image without one has
      nothing to teach.
    - `metadata` -- occurrence columns to carry into the manifest.
    - `metrics` -- metric run names whose values to carry into the manifest.
    - `require_mask` -- False exports occurrences with no mask, for training
      on whole images.
    - `subset`, `limit` -- narrow an unsplit export; rejected alongside
      `splits`.
    - `from_part` -- build each image from an upstream part's CANONICAL mask
      instead of `part`'s own; see `iterate_segments()`. Pass the same
      `from_part` the `run_segments()` call that made `part`'s mask used, so
      the exported image matches the shared crop the model will see at
      inference rather than one cropped to `part`'s own, usually much
      smaller, mask.
    - `visualize` -- True (default), an int, or ids: a pipeline grid per
      split of what was written, each image with its mask and class, and
      with any transform's own panels beside it. Named for the dataset's
      `data_hash`. False writes nothing.

    Returns the manifest DataFrame, one row per written image.
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
    reports = []

    for split_name, occurrence_ids in selections:
        # The dataset's own hash is only known once it's written, so each
        # report is named for it just before closing (see below).
        report = pipeline_visualization.open_report(
            project_path, f"dataset__{split_name or UNSPLIT}", "pending", part=part,
            visualize=visualize)
        reports.append(report)

        for occurrence_id, segment in iterate_segments(
                project_path, part=part, transforms=transforms,
                reference=reference, occurrence_ids=occurrence_ids,
                require_mask=require_mask, from_part=from_part, report=report):

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

            if segment.panel_sink is not None:
                segment.emit_panel(
                    segment_panel(segment.image, segment.mask,
                                  lines=[class_value] if class_value else []),
                    "exported")

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
    record = _write_dataset_record(output_dir, manifest, splits, part, transforms,
                                   reference, masks, class_by, from_part)

    identity = {key: value for key, value in record.items() if key != "created_at"}
    for report in reports:
        report.identify(record["data_hash"], identity=identity).close()

    logger.info("exported %d image(s) -> %s", len(manifest), output_dir)
    return manifest


def write_dataset(project_path, output_dir, part=DEFAULT_PART, transforms=(),
                  reference=False, subset=None, limit=None, label_columns=None,
                  label_runs=None, visualize=True):
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

    - `label_columns` -- occurrence columns to carry into the manifest
      (export_training_data's `metadata`).
    - `label_runs` -- metric run names to carry into the manifest
      (export_training_data's `metrics`).
    - `visualize` -- as in export_training_data.

    Everything else is as iterate_segments(). Returns the manifest DataFrame.
    """
    return export_training_data(
        project_path, output_dir, part=part, transforms=transforms,
        reference=reference, masks=True, metadata=label_columns,
        metrics=label_runs, subset=subset, limit=limit, visualize=visualize)


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
                          reference, masks, class_by, from_part=None):
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
        "from_part": from_part,
        "reference": bool(reference),
        "masks": bool(masks),
        "class_by": class_by,
        "classes": (sorted(manifest["class"].unique().tolist())
                    if class_by is not None else None),
        "transforms": [operation.spec() for operation in transforms],
        "splits": {name: ids_record(ids) for name, ids in sorted(groups.items())},
    }
    record["data_hash"] = hash_spec(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_json(os.path.join(output_dir, DATASET_FILE), record)
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
