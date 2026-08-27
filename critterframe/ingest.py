"""
Ingest: bringing external occurrence metadata and local images into a CritterFrame
project's normalized representation

The source file itself is archived into /imports under a dated name

Whatever a source calls its identifier and image URL column, a CritterFrame project calls them occurrence_id
and image_url

Source-specific ingestion (an Antenna export's column names, an iNaturalist API
response) lives in extensions/, and normalizes into these functions rather than
around them.

Almost everything a source sends comes through untouched -- normalization means
agreeing on two column names, not editing the data. The one exception is `drop`
(see ingest_occurrences), which excludes rows the source has already declared
are not organisms. That is the occurrence table's contract being enforced rather
than a filter being applied, and it is safe to enforce here precisely because
the source file is archived BEFORE it is parsed: imports/ keeps every row, so
what was excluded is always recoverable and always auditable.
"""

import logging
from datetime import date
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
    Copy a source file into the project's dated, immutable import archive.
    Returns the archive path.

    source_path -- path to the file being ingested.
    project_path -- project whose imports/ directory receives it.
    name_prefix -- distinguishes import kinds and usually carries the source,
                   e.g. "occurrences_antenna_199" ->
                   occurrences_antenna_199_2026-08-07.csv. A numeric suffix is
                   appended if that dated name already exists, so same-day
                   re-imports don't clobber each other rather than one silently
                   winning.
    """
    source_path = Path(source_path)
    imports_dir = paths.imports_dir(project_path)
    imports_dir.mkdir(parents=True, exist_ok=True)

    stamp = date.today().isoformat()
    extension = source_path.suffix or ".csv"
    base = f"{name_prefix}_{stamp}"

    dest = imports_dir / f"{base}{extension}"
    n = 1
    while dest.exists():
        dest = imports_dir / f"{base}_{n}{extension}"
        n += 1

    dest.write_bytes(source_path.read_bytes())
    logger.info("archived import -> %s", dest)
    return dest


def ingest_occurrences(project_path, import_csv_path, id_col=None,
                       image_url_col=None, datetime_cols=(), numeric_cols=(),
                       transform=None, drop=None, name_prefix="occurrences"):
    """
    Ingest an occurrence table from a CSV, as a full SNAPSHOT.

    The CSV states what the project's occurrences are, in full, and replaces
    whatever was there. To add occurrences, add rows to the CSV and re-ingest
    it -- don't ingest a second CSV of just the new ones.

    Re-ingesting is safe for work already done. Masks and metrics are keyed by
    occurrence id, so occurrences still present keep everything derived from
    them, and a re-ingest that adds rows leaves the next run with only the new
    ones to process.

    project_path    -- project to ingest into. The directory doesn't have to
                      exist: a project comes into being lazily, as its first
                      writer creates what it needs, and this is normally that
                      first writer.
    import_csv_path -- source CSV. Archived to project_path/imports/ before
                      being read.
    id_col          -- source column identifying an occurrence. Omit if it's
                      already called occurrence_id. Required, unlike the image
                      column: every occurrence needs an id, and duplicates or
                      blanks raise rather than being quietly dropped.
    image_url_col   -- source column holding the image URL. OPTIONAL. Omit it
                      if the source has no URLs because the images are local --
                      ingest the metadata here and the pixels with
                      ingest_images(), or pass the metadata straight to
                      ingest_images(metadata=...) so one snapshot carries both.
    datetime_cols,
    numeric_cols    -- source columns to coerce. Listing a column the source
                      didn't include is harmless.
    transform       -- optional callable(df) -> df applied after normalization,
                      for deriving columns a project needs. Extensions use this
                      rather than adding parameters here for every source's
                      quirks.
    drop            -- {column: values} naming rows the source has already said
                      are NOT organisms, e.g.
                      {"determination_name": ["Not Lepidoptera"]}. Those rows
                      don't reach the occurrence table.

                      This is the occurrence contract, not a filter. Every row
                      here asserts "one focal organism, at a place and time, in
                      this image" -- a detection the source itself classified as
                      debris asserts nothing, and a table mixing the two has
                      stopped being a statement about organisms at all. Which is
                      why it happens at ingest and why it's allowed to, when
                      nothing else in the package deletes a row: what's excluded
                      is a fact the SOURCE reported, not a judgement this project
                      made and might revise.

                      Not for quality. "Blurred", "score below 0.5", "probably
                      mis-segmented" are all judgements you will want to revisit
                      with a different threshold, and revisiting them must not
                      require re-ingesting -- use export filters or a subset,
                      which keep the data. The rule vocabulary is membership-only
                      so this one can't quietly become that one (see
                      selectionhelpers.rows_matching).

                      Applied after `transform`, so a rule may name a column the
                      transform joined in. Missing values never match: a row the
                      source didn't classify is not a row it classified as
                      debris. A named column the table doesn't have raises.
                      Every dropped row remains in the archived import.
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
    Ingest a folder of local images as a full SNAPSHOT: copy them into the
    project's image store and write one occurrence row per file.

    For projects whose images never came from a URL -- a folder of specimen
    photographs, frames pulled off an experimental camera, a hand-picked
    validation set. It's the local-file counterpart to
    ingest_occurrences + download_images.

    The FOLDER is the source of truth, the same way a CSV is for
    ingest_occurrences. Add images to it and re-ingest, and the project gains
    them; the occurrence table is rewritten from what the folder now contains
    rather than accumulated across calls. Every matched file is re-read and
    re-stored on each ingest, which is what makes replacing a file with a
    corrected version work -- the store follows the folder rather than keeping
    the first version it ever saw.

    No image_url is written, and none is needed: URLs exist so
    download_images() can fetch pixels, and these pixels are already here.

    Two consequences worth knowing. A file REMOVED from the folder loses its
    occurrence row on the next ingest, though its image, masks, and metrics
    stay in the project keyed by an id nothing now references. And this
    replaces the occurrence table, so a project combining a metadata CSV with a
    local image folder should pass that CSV as `metadata` here rather than
    ingesting it separately -- one snapshot, carrying both.

    One image is one occurrence, which is the same rule as everywhere else:
    each file must show one focal organism. A folder of multi-organism sheets
    has to be detected and cut into per-organism images upstream of this.

    Files are copied into the store BYTE FOR BYTE, in whatever format they
    already are. A 16-bit TIFF stays a 16-bit TIFF; a PNG with an alpha channel
    keeps it; EXIF survives. The image is decoded once here to check it's
    readable and to record its dimensions, and that decoded array is then
    discarded -- storing it instead would flatten every image to 8-bit BGR and
    re-encode it, irreversibly (see storage.imagestore).

    What gets archived here is a MANIFEST, not the pixels. Copying every image
    into imports/ would double a project's largest storage cost to duplicate
    files that are already on disk and already going into the image store; the
    manifest records each occurrence id, its source path, and the file's size
    and modification time, which is what you'd actually consult later to answer
    "where did this image come from and has it changed".

    project_path -- project to ingest into; created lazily, as above.
    image_dir    -- directory of images.
    patterns     -- glob patterns to match.
    id_from_stem -- derive each occurrence id from the filename stem. The only
                    supported scheme, and the flag exists to make it explicit
                    that ids come from filenames -- meaning filenames have to
                    be unique and stable, since renaming a file later would
                    orphan every mask and metric keyed to its old name.
    metadata     -- optional DataFrame of extra occurrence columns joined on
                    occurrence_id, whose values must match the filename stems.
                    This is how a metadata CSV and a folder of images become
                    one project: read the CSV yourself and pass it here.
    recursive    -- search subdirectories too.
    batch_size   -- images written to the store per transaction. Batched
                    because one LMDB transaction per image is markedly slower
                    than one per batch; flushed periodically rather than kept
                    in memory until the end so a very large folder doesn't
                    hold every file's bytes at once.

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
