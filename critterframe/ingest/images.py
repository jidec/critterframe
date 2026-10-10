"""Ingest a folder of local images: bytes into the image store, one occurrence row per file."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..core import drivers
from ..core.recipes import hash_spec
from ..records import occurrences as occurrence_records
from ..storage.imagestore import ImageStore
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate
from .archive import archive_raw_import, archived_hashes

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEFAULT_BATCH_SIZE = 100


def ingest_images(
    project_path,
    image_dir,
    patterns=DEFAULT_IMAGE_PATTERNS,
    id_from_stem=True,
    metadata=None,
    recursive=False,
    batch_size=DEFAULT_BATCH_SIZE,
    visualize=True,
    visualize_every=None,
):
    """Ingest a folder of local images as a full snapshot.

    Each file is copied byte for byte into the image store and becomes one occurrence
    row. Every matched file is read and stored again on each call. A file removed from
    the folder loses its row, while its image, masks and metrics stay.

    Args:
        project_path: Project to ingest into; created if absent.
        image_dir: Directory of images.
        patterns: Glob patterns to match.
        id_from_stem: Take each occurrence id from the filename stem. The only supported
            scheme, so stems must be unique and stable.
        metadata: DataFrame of extra occurrence columns, joined on `occurrence_id`.
        recursive: Search subdirectories.
        batch_size: Images written per store transaction.
        visualize: True, an int, or ids: a pipeline grid of thumbnails of what was ingested.
        visualize_every: Also write a thumbnail grid every N files.

    Returns:
        The `core.drivers.Tally.summary` dict plus `occurrences`, the rows in the resulting table.
    """
    if not id_from_stem:
        raise ValueError("id_from_stem=False isn't supported -- ids come from filenames")

    directory = Path(image_dir)
    glob = directory.rglob if recursive else directory.glob
    image_paths = sorted({path for pattern in patterns for path in glob(pattern)})

    _check_unique_stems(image_paths)

    stems = [path.stem for path in image_paths]
    identity = {
        "kind": "ingest_images",
        "image_dir": str(directory),
        "patterns": list(patterns),
        "recursive": recursive,
        "files": occurrence_records.ids_record(stems),
    }
    report = pipeline_visualization.open_report(
        project_path,
        f"ingest_images__{directory.name}",
        hash_spec(identity),
        visualize=visualize,
        visualize_every=visualize_every,
        identity=identity,
    ).begin(stems)

    tally = drivers.Tally(attempted=len(image_paths))
    rows = []
    batch = []

    with ImageStore(project_path) as store:
        for path in image_paths:
            try:
                data = path.read_bytes()

                # Decoded only to validate and to measure; IMREAD_UNCHANGED so
                # the dimensions come from the file as it really is rather than
                # from a converted copy of it. The array is not what gets
                # stored -- `data` is.
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("could not decode")

                occurrence_id = path.stem
                if report.wants(occurrence_id):
                    thumbnail = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if thumbnail is not None:
                        extension = path.suffix.lower().lstrip(".")
                        annotate(thumbnail, f"{image.shape[1]}x{image.shape[0]} {extension}")
                        report.panel(occurrence_id, "ingested", thumbnail)
                batch.append((occurrence_id, data))
                if len(batch) >= batch_size:
                    store.put_many(batch)
                    batch.clear()

                stat = path.stat()
                rows.append(
                    {
                        occurrence_records.ID_COL: occurrence_id,
                        "source_path": str(path),
                        "source_format": path.suffix.lower().lstrip("."),
                        "image_width": image.shape[1],
                        "image_height": image.shape[0],
                        "source_bytes": stat.st_size,
                        "source_mtime": pd.Timestamp(stat.st_mtime, unit="s"),
                    }
                )
            except Exception as exc:
                logger.warning("image ingest failed for %s: %s", path, exc)
                tally.record_failure(path.stem, exc, path=str(path))
                report.failure(path.stem, f"{path}: {exc}")

            report.done(path.stem)

        if batch:
            store.put_many(batch)
    report.close()

    if not rows:
        logger.warning("no images ingested from %s", image_dir)
        return tally.summary(occurrences=0)

    df = pd.DataFrame(rows)
    _archive_manifest(project_path, df, directory)

    if metadata is not None:
        df = df.merge(metadata, on=occurrence_records.ID_COL, how="left")

    table = occurrence_records.save_occurrences(project_path, df)

    tally.processed = len(rows)
    logger.info(
        "image ingest complete: attempted=%d saved=%d failed=%d",
        tally.attempted,
        tally.processed,
        tally.failed,
    )
    return tally.summary(occurrences=len(table))


def _check_unique_stems(image_paths):
    """Raise if two files would produce the same occurrence id.

    Checked before anything is written, and reported as the colliding paths.
    """
    by_stem = {}
    for path in image_paths:
        by_stem.setdefault(path.stem, []).append(path)

    collisions = {stem: paths_ for stem, paths_ in by_stem.items() if len(paths_) > 1}
    if not collisions:
        return

    detail = "; ".join(
        f"{stem}: {', '.join(str(p) for p in paths_)}" for stem, paths_ in sorted(collisions.items())[:3]
    )
    more = f" (and {len(collisions) - 3} more)" if len(collisions) > 3 else ""
    raise ValueError(
        f"{len(collisions)} filename stem(s) map to more than one image, so "
        f"they'd share an occurrence id -- {detail}{more}. Rename them so each "
        "image has a unique stem."
    )


def _archive_manifest(project_path, df, directory):
    """Archive the manifest of an image ingest in `raw_imports/`, reusing a byte-identical one."""
    name_prefix = f"images_{directory.name}"
    archive_raw_import(
        project_path,
        df.to_csv(index=False).encode("utf-8"),
        name_prefix,
        ".csv",
        archived_hashes(project_path, name_prefix),
    )
