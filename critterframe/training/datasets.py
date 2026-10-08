"""export_training_data(): images, masks, class folders and a manifest."""

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


def export_training_data(
    project_path,
    output_dir,
    splits=None,
    part=DEFAULT_PART,
    transforms=(),
    reference=False,
    masks=False,
    class_by=None,
    metadata=None,
    metrics=None,
    require_mask=True,
    subset=None,
    limit=None,
    from_part=None,
    visualize=True,
):
    """Write project data as a directory a trainer can read.

    The split level exists only with `splits`, the class level only with `class_by`:

        output_dir/
            manifest.csv
            dataset.json
            train/<class>/<occurrence_id>.png     (class_by given)
            train/images/<occurrence_id>.png      (no class_by)
            train/masks/<occurrence_id>.png       (masks=True)

    Args:
        project_path: Project to export from.
        output_dir: Directory to write into; created if missing.
        splits: `{split name: subset name or occurrence ids}`, e.g. what `split_ids` returned.
            None exports one flat dataset. An occurrence in two splits raises.
        part: Part to export.
        transforms: Operations applied to each segment before writing; use the chain the
            model will get at inference.
        reference: Read masks from the reference table.
        masks: Also write each mask as a 0/255 PNG.
        class_by: Occurrence column to sort images into class folders by. An occurrence
            with no value is left out.
        metadata: Occurrence columns to carry into the manifest.
        metrics: Metric run names whose values to carry into the manifest.
        require_mask: False also exports occurrences with no mask.
        subset: Named subset, for an unsplit export.
        limit: Cap on occurrences, for an unsplit export.
        from_part: Frame each image by this upstream part's mask, as the `run_segments` call
            that made `part` did.
        visualize: True, an int, or ids: a pipeline grid per split of what was written.

    Returns:
        The manifest DataFrame, one row per written image.
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
            project_path, f"dataset__{split_name or UNSPLIT}", "pending", part=part, visualize=visualize
        )
        reports.append(report)

        for occurrence_id, segment in iterate_segments(
            project_path,
            part=part,
            transforms=transforms,
            reference=reference,
            occurrence_ids=occurrence_ids,
            require_mask=require_mask,
            from_part=from_part,
            report=report,
            progress=f"export_training_data split '{split_name or UNSPLIT}'",
        ):
            class_value = None
            if class_by is not None:
                class_value = class_values.get(occurrence_id)
                if class_value is None:
                    unclassed += 1
                    continue

            leaf = _class_folder(folders, class_value) if class_by else IMAGE_DIR
            image_path = os.path.join(_directory(output_dir, split_name, leaf), f"{occurrence_id}.png")
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
                        _directory(output_dir, split_name, MASK_DIR), f"{occurrence_id}.png"
                    )
                    # 0/255 PNG: lossless, so a mask boundary survives the
                    # round trip exactly, and readable by anything.
                    _write_image(mask_path, segment.mask.astype("uint8") * 255)
                    mask_path = _relative(mask_path, output_dir)
                row["mask_path"] = mask_path

            row["height"] = segment.shape[0]
            row["width"] = segment.shape[1]
            row["mask_area"] = None if segment.mask is None else int(segment.mask.sum())
            rows.append(row)

            if segment.panel_sink is not None:
                segment.emit_panel(
                    segment_panel(segment.image, segment.mask, lines=[class_value] if class_value else []),
                    "exported",
                )

    if unclassed:
        logger.info("left out %d occurrence(s) with no value in '%s'", unclassed, class_by)
    if unwritten:
        logger.warning(
            "%d image(s) could not be written -- check the "
            "occurrence ids for characters the filesystem rejects",
            unwritten,
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        logger.warning(
            "nothing exported -- does part '%s' have %s masks?",
            part,
            "reference" if reference else "canonical",
        )
        return manifest

    manifest = _attach_labels(project_path, manifest, metadata, metrics, part)
    manifest.to_csv(os.path.join(output_dir, MANIFEST_FILE), index=False)
    record = _write_dataset_record(
        output_dir, manifest, splits, part, transforms, reference, masks, class_by, from_part
    )

    identity = {key: value for key, value in record.items() if key != "created_at"}
    for report in reports:
        report.identify(record["data_hash"], identity=identity).close()

    logger.info("exported %d image(s) -> %s", len(manifest), output_dir)
    return manifest


def write_dataset(
    project_path,
    output_dir,
    part=DEFAULT_PART,
    transforms=(),
    reference=False,
    subset=None,
    limit=None,
    label_columns=None,
    label_runs=None,
    visualize=True,
):
    """Write an image and mask per occurrence plus a manifest, in one flat directory.

    Args:
        project_path: Project to export from.
        output_dir: Directory to write into.
        part: Part to export.
        transforms: Operations applied to each segment before writing.
        reference: Read masks from the reference table.
        subset: Named subset to restrict to.
        limit: Cap on occurrences.
        label_columns: Occurrence columns to carry into the manifest.
        label_runs: Metric run names whose values to carry into the manifest.
        visualize: As in `export_training_data`.

    Returns:
        The manifest DataFrame.
    """
    return export_training_data(
        project_path,
        output_dir,
        part=part,
        transforms=transforms,
        reference=reference,
        masks=True,
        metadata=label_columns,
        metrics=label_runs,
        subset=subset,
        limit=limit,
        visualize=visualize,
    )


def _resolve_splits(project_path, splits, subset, limit):
    """Return `splits` as an ordered `[(split name, ids)]`, raising if an occurrence is in two."""
    if splits is None:
        return [(None, subset_selection.select_ids(project_path, subset=subset, limit=limit))]

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

    logger.info(
        "exporting %d split(s): %s", len(resolved), ", ".join(f"{name}={len(ids)}" for name, ids in resolved)
    )
    return resolved


def _class_values(project_path, class_by):
    """Return `{occurrence_id: class value}`, leaving out missing and blank values."""
    if class_by is None:
        return {}

    occurrences = subset_selection.select_occurrences(project_path, columns=[class_by])
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
    """Return the folder name for a class.

    Raises if two classes would share one: `Anax junius` and `Anax/junius` both sanitize
    to `Anax_junius`.
    """
    if class_value in folders:
        return folders[class_value]

    folder = _UNSAFE_IN_NAME.sub("_", class_value).strip("_") or "unnamed"
    clash = next((existing for existing, name in folders.items() if name == folder), None)
    if clash is not None:
        raise ValueError(
            f"classes {clash!r} and {class_value!r} both become folder "
            f"'{folder}' -- rename one in the occurrence table, or export with "
            "a different class_by column"
        )

    folders[class_value] = folder
    return folder


def _directory(output_dir, split_name, leaf):
    """Return the directory one image belongs in, creating it."""
    parts = [output_dir] + ([split_name] if split_name else []) + [leaf]
    directory = os.path.join(*parts)
    os.makedirs(directory, exist_ok=True)
    return directory


def _relative(path, output_dir):
    """Return a written file's path relative to the dataset directory, with forward slashes.

    Forward slashes because a dataset is often moved from Windows to a Linux cluster.
    """
    return Path(os.path.relpath(path, output_dir)).as_posix()


def _write_image(path, image):
    """Write one PNG, returning whether it succeeded.

    `cv2.imwrite` returns False instead of raising on a path the filesystem refuses.
    """
    if cv2.imwrite(path, image):
        return True
    logger.warning("could not write %s", path)
    return False


def _write_dataset_record(
    output_dir, manifest, splits, part, transforms, reference, masks, class_by, from_part=None
):
    """Write `dataset.json`: the export described from what was written, with its `data_hash`."""
    if splits is None:
        groups = {UNSPLIT: manifest["occurrence_id"].tolist()}
    else:
        groups = {name: frame["occurrence_id"].tolist() for name, frame in manifest.groupby("split")}

    record = {
        "part": part,
        "from_part": from_part,
        "reference": bool(reference),
        "masks": bool(masks),
        "class_by": class_by,
        "classes": (sorted(manifest["class"].unique().tolist()) if class_by is not None else None),
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
        occurrences = subset_selection.select_occurrences(project_path, columns=list(label_columns))
        manifest = manifest.merge(occurrences, on="occurrence_id", how="left")

    if label_runs:
        values = metrics_wide(project_path, run_names=list(label_runs), parts=[part])
        if not values.empty:
            manifest = manifest.merge(values, on="occurrence_id", how="left")

    return manifest
