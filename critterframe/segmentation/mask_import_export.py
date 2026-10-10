"""import_masks() and export_masks(): masks in and out of a project as `<occurrence_id>__<part>.png` files."""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..core import drivers
from ..project import paths
from ..selection import subsets as subset_selection
from ..core.recipes import DEFAULT_PART, Recipe, Segmentation, hash_spec
from ..records import masks as mask_records
from ..records import occurrences as occurrence_records
from ..records import runs as run_records
from ..records.occurrences import ID_COL, ids_record
from ..storage.imagestore import ImageStore
from ..storage.jsonfiles import write_json
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate, overlay_mask

logger = logging.getLogger(__name__)

MANIFEST_NAME = "masks.export.json"
PART_SEPARATOR = "__"

# A pixel above this is mask, so a brush's anti-aliased edge binarizes the obvious way.
MASK_CUTOFF = 127

# Characters a file name can't carry on at least one of Windows, Linux and macOS.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

BATCH_SIZE = 250


def import_masks(
    project_path,
    folder,
    part=DEFAULT_PART,
    run_name=None,
    description=None,
    reference=False,
    replace=True,
    visualize=True,
):
    """Import a folder of mask PNGs named `<occurrence_id>__<part>.png`.

    Each mask must match its occurrence's original image in size; a pixel above 127 is
    mask. One run is recorded per part, and a `masks.export.json` in the folder is kept
    on it as the masks' source.

    Args:
        project_path: Project to import into.
        folder: Folder of mask PNGs.
        part: Part for a file named `<occurrence_id>.png`, with no part in its name.
        run_name: Name for the run; `<part>_imported` if None, or
            `<part>_reference_imported` with `reference`.
        description: Free text saying what the masks are; recorded on the run.
        reference: Write to the reference table.
        replace: Replace an occurrence-part's existing mask; False keeps it.
        visualize: Write a pipeline grid of imported masks over their images.

    Returns:
        `{part: summary}`, each a `core.drivers.Tally.summary` plus `run_id`, `replaced`,
        `unmatched` (files naming no occurrence here) and `unverified` (masks with no
        image to check their size against).
    """
    paths.require_project(project_path)
    folder = Path(folder)
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    known = set(occurrence_records.load_occurrences(project_path, columns=[])[ID_COL])
    source = _read_manifest(folder)

    by_part, unmatched = {}, Counter()
    for path in files:
        occurrence_id, file_part = _split_name(path.stem, known, part)
        if occurrence_id not in known:
            unmatched[file_part] += 1
            continue
        by_part.setdefault(file_part, []).append((occurrence_id, path))
    if unmatched:
        logger.warning(
            "import_masks: %d file(s) in %s name no occurrence in this project",
            sum(unmatched.values()),
            folder.name,
        )

    store = (
        ImageStore(project_path, readonly=True)
        if (paths.images_path(project_path) / "data.mdb").exists()
        else None
    )
    results = {}
    try:
        for file_part, entries in sorted(by_part.items()):
            results[file_part] = _import_part(
                project_path,
                file_part,
                entries,
                store,
                folder,
                source,
                run_name,
                description,
                reference,
                replace,
                visualize,
                unmatched.pop(file_part, 0),
            )
    finally:
        if store is not None:
            store.close()

    for file_part, count in unmatched.items():
        results[file_part] = drivers.Tally().summary(run_id=None, replaced=0, unmatched=count, unverified=0)
    return results


def _import_part(
    project_path,
    part,
    entries,
    store,
    folder,
    source,
    run_name,
    description,
    reference,
    replace,
    visualize,
    unmatched,
):
    """Import one part's files as one run, and return its summary."""
    default = f"{part}_reference_imported" if reference else f"{part}_imported"
    recipe = Recipe(
        "segment",
        run_name or default,
        [Segmentation("import_masks", _cannot_recompute, {}, version="1")],
        part=part,
        inputs={"masks": "reference" if reference else "canonical"},
    )
    ids = [occurrence_id for occurrence_id, _path in entries]
    existing = mask_records.mask_lookup(project_path, part=part, occurrence_ids=ids, reference=reference)
    run_id = run_records.start_run(
        project_path,
        recipe,
        context={
            "occurrences": ids_record(ids),
            "folder": folder.name,
            "description": description,
            "source": source,
        },
    )
    report = pipeline_visualization.open_report(
        project_path, recipe.name, recipe.hash, part=part, visualize=visualize, identity=recipe.spec()
    ).begin(ids)

    tally = drivers.Tally(attempted=len(entries))
    rows, replaced, unverified = [], 0, 0
    for occurrence_id, path in entries:
        try:
            mask = _read_mask(path)
            image = store.get(occurrence_id) if store is not None else None
            if image is None:
                unverified += 1
            elif image.shape[:2] != mask.shape:
                raise ValueError(
                    f"mask is {mask.shape[1]}x{mask.shape[0]} but the image is "
                    f"{image.shape[1]}x{image.shape[0]} -- a mask must be in the "
                    "original image's coordinates"
                )

            row_hash = hash_spec({"recipe": recipe.hash, "mask": mask_records.mask_digest(mask)})
            current = existing.get(occurrence_id)
            if current is not None and (current["recipe_hash"] == row_hash or not replace):
                tally.skipped += 1
            else:
                replaced += current is not None
                rows.append(
                    mask_records.make_mask_row(
                        occurrence_id,
                        mask,
                        part=part,
                        recipe_hash=row_hash,
                        run_id=run_id,
                        info={"import_masks": {"file": path.name}},
                    )
                )
                tally.processed += 1
                if image is not None and report.wants(occurrence_id):
                    panel = overlay_mask(image, mask)
                    annotate(panel, path.name)
                    report.panel(occurrence_id, "imported", panel)
            if len(rows) >= BATCH_SIZE:
                mask_records.save_masks(project_path, rows, reference=reference)
                rows = []
        except Exception as exc:
            tally.record_failure(occurrence_id, exc, file=path.name)
            report.failure(occurrence_id, exc)
        report.done(occurrence_id)

    report.close()
    if rows:
        mask_records.save_masks(project_path, rows, reference=reference)
    run_records.finish_run(
        project_path, run_id, processed=tally.processed, skipped=tally.skipped, failed=tally.failed
    )
    if unverified:
        logger.warning(
            "import_masks part '%s': %d mask(s) had no image to check their size against", part, unverified
        )
    return tally.summary(run_id=run_id, replaced=replaced, unmatched=unmatched, unverified=unverified)


def export_masks(project_path, dest, parts=None, subset=None, reference=False):
    """Write masks as `<occurrence_id>__<part>.png` files, plus a `masks.export.json`.

    Each is a single-channel 0/255 PNG in the original image's coordinates. Nothing is
    written if a name can't be a file on every OS or already exists in `dest`.

    Args:
        project_path: Project to export from.
        dest: Folder to write into; created if absent.
        parts: Parts to export; all if None.
        subset: Named subset to restrict to.
        reference: Export reference masks.

    Returns:
        `{part: summary}`, with `processed` the files written, plus `directory`.
    """
    paths.require_project(project_path)
    ids = subset_selection.select_ids(project_path, subset=subset) if subset else None
    df = mask_records.load_masks(project_path, parts=parts, occurrence_ids=ids, reference=reference)
    if df.empty:
        logger.warning("export_masks: no masks to export")
        return {}

    rows = df.to_dict("records")
    names = [paths.product_filename(row["occurrence_id"], row["part"]) for row in rows]
    _check_names(rows, names)
    dest = Path(dest)
    clashes = [name for name in names + [MANIFEST_NAME] if (dest / name).exists()]
    if clashes:
        raise FileExistsError(f"{len(clashes)} file(s) already in {dest}, e.g. {clashes[:3]}")
    dest.mkdir(parents=True, exist_ok=True)

    files = {}
    for row, name in zip(rows, names):
        mask = mask_records.decode_mask(row)
        ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise ValueError(f"could not encode {name}")
        (dest / name).write_bytes(encoded.tobytes())
        files[name] = {"recipe_hash": row["recipe_hash"], "digest": mask_records.mask_digest(mask)}

    counts = Counter(row["part"] for row in rows)
    write_json(
        dest / MANIFEST_NAME,
        {
            "project": paths.project_dir(project_path).resolve().name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reference": bool(reference),
            "parts": dict(sorted(counts.items())),
            "runs": _source_runs(project_path, rows),
            "files": files,
        },
    )
    logger.info("exported %d mask(s) -> %s", len(rows), dest)

    results = {}
    for part, count in sorted(counts.items()):
        tally = drivers.Tally(attempted=count)
        tally.processed = count
        results[part] = tally.summary(directory=str(dest))
    return results


def _cannot_recompute(segment):
    raise ValueError("an imported mask can't be recomputed -- import its file again")


def _split_name(stem, known_ids, part):
    """Return `(occurrence_id, part)` from a file stem.

    Splits at the last `__`, unless the whole stem is a known id and the split is not.
    """
    if PART_SEPARATOR in stem:
        occurrence_id, named_part = stem.rsplit(PART_SEPARATOR, 1)
        if occurrence_id in known_ids or stem not in known_ids:
            return occurrence_id, named_part
    return stem, part


def _read_mask(path):
    """Read a mask file as a boolean array, including from a non-ASCII path on Windows."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("not a readable image")
    mask = image > MASK_CUTOFF
    if not mask.any():
        raise ValueError("empty mask")
    return mask


def _read_manifest(folder):
    """Return what an `export_masks` manifest in the folder says about the masks' source, or None."""
    path = folder / MANIFEST_NAME
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return {key: record.get(key) for key in ("project", "created_at", "reference", "runs")}


def _check_names(rows, names):
    """Raise if a name can't be a file on every OS."""
    bad = [
        name
        for row, name in zip(rows, names)
        if _UNSAFE.search(name)
        or str(row["occurrence_id"]).endswith((".", " "))
        or PART_SEPARATOR in str(row["part"])
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} mask(s) can't be written as a file on every OS "
            f"(or have a part containing '__'), e.g. {bad[:3]}"
        )
    folded = Counter(name.lower() for name in names)
    same = sorted(name for name in names if folded[name.lower()] > 1)
    if same:
        raise ValueError(
            f"file names differing only by case would overwrite each other on Windows and macOS: {same[:4]}"
        )


def _source_runs(project_path, rows):
    """Return the distinct runs the masks came from: run id, name and recipe hash."""
    run_ids = {int(row["run_id"]) for row in rows if not pd.isna(row.get("run_id"))}
    runs = run_records.load_runs(project_path)
    if runs.empty or not run_ids:
        return []
    kept = runs[runs["run_id"].isin(run_ids)].sort_values("run_id")
    return [
        {"run_id": int(run["run_id"]), "name": run["name"], "recipe_hash": run["recipe_hash"]}
        for run in kept.to_dict("records")
    ]
