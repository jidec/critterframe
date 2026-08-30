"""
Ingest occurrence tables and optionally local images.

Both forms are full SNAPSHOTS: the source states what the project's occurrences
are and replaces what was there, so to add data you extend the source and
re-ingest. The source file is archived into imports/ before it is parsed, which
is what makes ingest_occurrences(drop=...) safe -- every dropped row stays
recoverable.

Source-specific ingestion lives in extensions/ and normalizes into these
functions rather than around them.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import selectionhelpers
from .project import paths
from .records import occurrences as occurrence_records
from .storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEFAULT_BATCH_SIZE = 100


def _archive_import(source_path, project_path, name_prefix):
    """
    Copy a source file into the project's dated, immutable import archive, and
    return the archive path.

    name_prefix -- import kind, usually carrying the source, e.g.
                   "occurrences_antenna_199". A numeric suffix is appended if
                   that dated name exists, so same-day re-imports don't clobber
                   each other.
    """
    source_path = Path(source_path)
    paths.imports_dir(project_path).mkdir(parents=True, exist_ok=True)

    dest = paths.import_path(project_path, name_prefix,
                             extension=source_path.suffix)
    dest.write_bytes(source_path.read_bytes())
    logger.info("archived import -> %s", dest)
    return dest


def ingest_occurrences(project_path, import_csv_path, id_col=None,
                       image_url_col=None, datetime_cols=(), numeric_cols=(),
                       transform=None, drop=None, name_prefix="occurrences"):
    """
    Ingest an occurrence table from a CSV, as a full snapshot.

    Re-ingesting is safe for work already done: masks and metrics are keyed by
    occurrence id, so occurrences still present keep everything derived from
    them.

    project_path    -- project to ingest into; created lazily if absent.
    import_csv_path -- source CSV, archived to imports/ before being read.
    id_col          -- source column identifying an occurrence. Omit if already
                       called occurrence_id. Duplicates or blanks raise.
    image_url_col   -- source column holding the image URL. Optional -- omit it
                       when the images are local.
    datetime_cols,
    numeric_cols    -- source columns to coerce. Naming an absent column is
                       harmless.
    transform       -- optional callable(df) -> df applied after normalization,
                       for deriving columns a project needs.
    drop            -- {column: values} naming rows the source has already said
                       are NOT organisms, e.g.
                       {"determination_name": ["Not Lepidoptera"]}.

                       This is the occurrence contract, not a filter: a row the
                       source classified as debris asserts no organism, so it
                       doesn't belong in a table whose every row does. Not for
                       quality -- "blurred" or "score below 0.5" are judgements
                       you will want to revise, so use export filters, which
                       keep the data. Missing values never match; an unknown
                       column raises; dropped rows stay in the archived import.
    name_prefix     -- prefix for the archived import's filename.

    Returns the resulting occurrence table.
    """
    _archive_import(import_csv_path, project_path, name_prefix)

    df = pd.read_csv(import_csv_path)
    logger.info("read %d rows from %s", len(df), import_csv_path)

    df = occurrence_records.normalize(df, id_col=id_col,
                                      image_col=image_url_col,
                                      datetime_cols=datetime_cols,
                                      numeric_cols=numeric_cols)
    if transform is not None:
        df = transform(df)

    if drop:
        excluded = selectionhelpers.rows_matching(df, drop)
        if excluded.any():
            logger.info("dropped %d of %d row(s) the source declared are not "
                        "organisms (%s); the archived import keeps all of them",
                        int(excluded.sum()), len(df), drop)
        df = df[~excluded].reset_index(drop=True)

    return occurrence_records.save_occurrences(project_path, df)


def ingest_images(project_path, image_dir, patterns=DEFAULT_IMAGE_PATTERNS,
                  id_from_stem=True, metadata=None, recursive=False,
                  batch_size=DEFAULT_BATCH_SIZE):
    """
    Ingest a folder of local images as a full snapshot: copy them into the image
    store and write one occurrence row per file.

    The folder is the source of truth. Every matched file is re-read and
    re-stored on each ingest, which is what makes replacing a file with a
    corrected version work. A file removed from the folder loses its occurrence
    row, though its image, masks and metrics stay behind under an id nothing
    references.

    Files are copied byte for byte, in whatever format they already are. The
    decode here only checks readability and records dimensions.

    What gets archived is a MANIFEST, not the pixels -- copying every image into
    imports/ would double a project's largest storage cost.

    project_path -- project to ingest into; created lazily.
    image_dir    -- directory of images.
    patterns     -- glob patterns to match.
    id_from_stem -- derive each occurrence id from the filename stem. The only
                    supported scheme, so filenames must be unique and stable --
                    renaming one orphans every mask and metric keyed to it.
    metadata     -- optional DataFrame of extra occurrence columns joined on
                    occurrence_id, whose values match the filename stems. How a
                    metadata CSV and a folder of images become one project.
    recursive    -- search subdirectories too.
    batch_size   -- images written to the store per LMDB transaction.

    Returns a summary dict (attempted, saved, failed, occurrences).
    """
    if not id_from_stem:
        raise ValueError("id_from_stem=False isn't supported -- ids come from filenames")

    directory = Path(image_dir)
    glob = directory.rglob if recursive else directory.glob
    image_paths = sorted({path for pattern in patterns for path in glob(pattern)})

    _check_unique_stems(image_paths)

    attempted = 0
    rows = []
    failures = []
    batch = []

    with ImageStore(project_path) as store:
        for path in image_paths:
            attempted += 1
            try:
                data = path.read_bytes()

                # Decoded only to validate and to measure; IMREAD_UNCHANGED so
                # the dimensions come from the file as it really is rather than
                # from a converted copy of it. The array is not what gets
                # stored -- `data` is.
                image = cv2.imdecode(np.frombuffer(data, np.uint8),
                                     cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("could not decode")

                occurrence_id = path.stem
                batch.append((occurrence_id, data))
                if len(batch) >= batch_size:
                    store.put_many(batch)
                    batch.clear()

                stat = path.stat()
                rows.append({
                    occurrence_records.ID_COL: occurrence_id,
                    "source_path": str(path),
                    "source_format": path.suffix.lower().lstrip("."),
                    "image_width": image.shape[1],
                    "image_height": image.shape[0],
                    "source_bytes": stat.st_size,
                    "source_mtime": pd.Timestamp(stat.st_mtime, unit="s"),
                })
            except Exception as exc:
                logger.warning("image ingest failed for %s: %s", path, exc)
                failures.append({"path": str(path), "error": str(exc)})

        if batch:
            store.put_many(batch)

    if not rows:
        logger.warning("no images ingested from %s", image_dir)
        return {"attempted": attempted, "saved": 0, "failed": len(failures),
                "failures": failures, "occurrences": 0}

    df = pd.DataFrame(rows)
    _archive_manifest(project_path, df, directory)

    if metadata is not None:
        df = df.merge(metadata, on=occurrence_records.ID_COL, how="left")

    table = occurrence_records.save_occurrences(project_path, df)

    logger.info("image ingest complete: attempted=%d saved=%d failed=%d",
                attempted, len(rows), len(failures))
    return {"attempted": attempted, "saved": len(rows), "failed": len(failures),
            "failures": failures, "occurrences": len(table)}


def _check_unique_stems(image_paths):
    """
    Raise if two files would produce the same occurrence id.

    Ids come from filename stems, so photo.jpg and photo.tiff collide, and so
    do a/frame1.png and b/frame1.png under recursive=True. Checked BEFORE
    anything is written, and reported as the colliding PATHS -- the generic
    duplicate-id error further downstream would name the id, which doesn't tell
    you which two files to go and rename.
    """
    by_stem = {}
    for path in image_paths:
        by_stem.setdefault(path.stem, []).append(path)

    collisions = {stem: paths_ for stem, paths_ in by_stem.items() if len(paths_) > 1}
    if not collisions:
        return

    detail = "; ".join(
        f"{stem}: {', '.join(str(p) for p in paths_)}"
        for stem, paths_ in sorted(collisions.items())[:3]
    )
    more = f" (and {len(collisions) - 3} more)" if len(collisions) > 3 else ""
    raise ValueError(
        f"{len(collisions)} filename stem(s) map to more than one image, so "
        f"they'd share an occurrence id -- {detail}{more}. Rename them so each "
        "image has a unique stem."
    )


def _archive_manifest(project_path, df, directory):
    """Write the manifest of an image ingest into the project's imports directory."""
    manifest_dir = paths.imports_dir(project_path)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    temporary = manifest_dir / ".manifest.csv"
    df.to_csv(temporary, index=False)
    _archive_import(temporary, project_path, f"images_{directory.name}")
    temporary.unlink()
