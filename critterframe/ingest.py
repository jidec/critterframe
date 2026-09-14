"""
Ingest occurrence tables and optionally local images.

A RAW IMPORT is a source's data exactly as it arrived, byte for byte, before
anything decided what any of it means. An IMPORT is what a raw import becomes
once the structural reshaping (which column is the id, which is the image url)
and the judgement calls (drop=, group_col/max_per_group) that turn it into
occurrences have been applied. ingest_occurrences archives the former into
raw_imports/ before it is parsed -- which is what makes drop= safe, every
dropped row stays recoverable -- and writes a manifest beside it recording
every decision that produced the latter, mirroring the manifest export_metrics
writes for what leaves a project.

Both ingest forms are full SNAPSHOTS: the source states what the project's
occurrences are and replaces what was there, so to add data you extend the
source and re-ingest. ingest_occurrences is also idempotent: re-running it on
the same raw content with the same decisions is recognized and skipped rather
than repeated, so a scheduled pull that finds nothing new does no work and
leaves no duplicate archive.

Source-specific ingestion lives in extensions/ and normalizes into these
functions rather than around them.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import selectionhelpers
from .project import paths
from .records import occurrences as occurrence_records
from .recipes import canonical_json, hash_spec
from .storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEFAULT_BATCH_SIZE = 100

def _content_hash(data):
    """A short, stable digest of raw bytes -- how a raw import is recognized again."""
    return hashlib.sha256(data).hexdigest()[:16]

def _recorded_callable(value):
    """
    A rule recorded by NAME rather than behaviour, the same reasoning
    export._recorded_filters gives: two different lambdas both called
    `first` are indistinguishable here, and a lambda's repr carries a memory
    address that would make one import hash differently every run.
    """
    if value is None or isinstance(value, str):
        return value
    return getattr(value, "__qualname__", "<callable>")


def _import_recipe(raw_hash, id_col, image_url_col, datetime_cols, numeric_cols,
                   transform, drop, group_col, max_per_group, cap_rule,
                   manifest_extra):
    """
    The dict import_hash is computed from. Shared between ingest_occurrences'
    own idempotency check and already_ingested(), so the two can never land
    on a different hash for the same inputs.
    """
    return {
        "raw_hash": raw_hash,
        "id_col": id_col,
        "image_url_col": image_url_col,
        "datetime_cols": sorted(datetime_cols),
        "numeric_cols": sorted(numeric_cols),
        "transform": _recorded_callable(transform),
        "drop": drop,
        "group_col": group_col,
        "max_per_group": max_per_group,
        "cap_rule": _recorded_callable(cap_rule),
        "extra": manifest_extra or {},
    }


def _fingerprint_raw_hash(existing_imports, source_path, stat):
    """
    The raw_hash of the most recent already-ingested import recorded against
    this exact source path, size, and mtime -- or None if nothing matches.
    """
    if existing_imports.empty or "import_source_path" not in existing_imports.columns:
        return None
    candidates = existing_imports[
        (existing_imports["import_source_path"] == str(source_path))
        & (existing_imports["import_source_bytes"] == stat.st_size)
        & (existing_imports["import_source_mtime"] == stat.st_mtime)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[-1]["raw_hash"]  # load_imports is oldest-first


def fingerprint_unchanged(project_path, source_path, id_col=None, image_url_col=None,
                          datetime_cols=(), numeric_cols=(), transform=None, drop=None,
                          group_col=None, max_per_group=None, cap_rule="random",
                          manifest_extra=None):
    """
    Whether source_path's path, size, and mtime match a previous import well
    enough to trust that import's own raw_hash instead of reading and
    hashing source_path again.

    NOT a content guarantee -- see ingest_occurrences' trust_source_file_unchanged
    for the risk this accepts. Every argument except source_path must be
    exactly what ingest_occurrences will be called with, same as
    already_ingested.

    Returns True if project_path's import log already has an import whose
    recorded source path/size/mtime match source_path today, and whose
    resulting import_hash -- recomputed from that import's own raw_hash plus
    these arguments -- matches an import already on record.
    """
    existing_imports = load_imports(project_path)
    if existing_imports.empty:
        return False
    try:
        stat = Path(source_path).stat()
    except OSError:
        return False
    borrowed_raw_hash = _fingerprint_raw_hash(existing_imports, source_path, stat)
    if borrowed_raw_hash is None:
        return False
    recipe = _import_recipe(borrowed_raw_hash, id_col, image_url_col, datetime_cols,
                            numeric_cols, transform, drop, group_col,
                            max_per_group, cap_rule, manifest_extra)
    import_hash = hash_spec(recipe)
    return bool((existing_imports["import_hash"] == import_hash).any())


def already_ingested(project_path, raw_bytes, id_col=None, image_url_col=None,
                     datetime_cols=(), numeric_cols=(), transform=None, drop=None,
                     group_col=None, max_per_group=None, cap_rule="random",
                     manifest_extra=None):
    """
    Whether this exact raw content, turned into an import by this exact set of
    decisions, has already been ingested -- computed from raw_bytes alone,
    with no parsed table required.

    ingest_occurrences already skips its own work on a repeat call, but only
    AFTER whatever parsed df it needs exists -- fine when that parse is
    pd.read_csv, not fine when a caller's own parse of raw_bytes is itself the
    expensive part (a multi-gigabyte source needing minutes and gigabytes of
    RAM to read, e.g. a GBIF Darwin Core Archive). Calling this FIRST, before
    doing that parse, is what makes a repeat ingest actually skip it rather
    than only skip the write that happens after it. See
    gbif_darwincore_inat.ingest.ingest_occurrences for the caller this exists
    for.

    Every argument except raw_bytes must be exactly what ingest_occurrences
    will be called with (typically via df=) -- these are hashed together, and
    a mismatched value here answers a question about a different import.

    Returns True if project_path's import log already has an import with
    this exact hash.
    """
    existing_imports = load_imports(project_path)
    if existing_imports.empty:
        return False
    raw_hash = _content_hash(raw_bytes)
    recipe = _import_recipe(raw_hash, id_col, image_url_col, datetime_cols,
                            numeric_cols, transform, drop, group_col,
                            max_per_group, cap_rule, manifest_extra)
    import_hash = hash_spec(recipe)
    return bool((existing_imports["import_hash"] == import_hash).any())


def load_imports(project_path):
    """
    Every import this project has produced from a raw import, oldest first.

    Reads the log ingest_occurrences appends to, the same pattern
    export.load_exports reads back. Returns a DataFrame; empty where this
    project has never ingested an occurrence table.
    """
    log = paths.imports_log_path(project_path)
    if not log.exists():
        return pd.DataFrame()

    with open(log, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(records)


def _archive_raw_import(project_path, data, name_prefix, extension, known_raw_hashes):
    """
    Write `data` into raw_imports/ as a new dated file, unless a raw import
    with identical content is already archived there -- in which case that
    file is reused rather than duplicated, since the same bytes are the same
    raw import no matter how many different decisions get made from them.

    known_raw_hashes -- {raw_hash: path string} of every raw import this
                        project's log already knows about (see load_imports).

    Returns (path, is_new).
    """
    raw_hash = _content_hash(data)
    existing = known_raw_hashes.get(raw_hash)
    if existing is not None and Path(existing).exists():
        logger.info("raw import is byte-identical to %s -- reusing it", existing)
        return Path(existing), False

    directory = paths.raw_imports_dir(project_path)
    directory.mkdir(parents=True, exist_ok=True)
    dest = paths.raw_import_path(project_path, name_prefix, extension=extension)
    dest.write_bytes(data)
    logger.info("archived raw import -> %s", dest)
    return dest, True


def _archive_source_file(source_path, project_path, name_prefix):
    """
    Copy a source file into the project's dated, immutable raw-import archive,
    and return the archive path.

    Used only by ingest_images' manifest archiving, which has no decisions to
    record beside it and so needs no import_hash/manifest machinery --
    ingest_occurrences uses _archive_raw_import instead.
    """
    source_path = Path(source_path)
    return _archive_raw_import(
        project_path, source_path.read_bytes(), name_prefix,
        source_path.suffix, known_raw_hashes={})[0]


def ingest_occurrences(project_path, import_csv_path, id_col=None,
                       image_url_col=None, datetime_cols=(), numeric_cols=(),
                       transform=None, drop=None, group_col=None,
                       max_per_group=None, cap_rule="random",
                       name_prefix="occurrences", raw_bytes=None,
                       raw_extension=None, manifest_extra=None, df=None,
                       trust_source_file_unchanged=False, fingerprint_source_path=None):
    """
    Turn a raw import into an occurrence table, as a full snapshot.

    Re-ingesting is safe for work already done as masks and metrics are keyed by
    occurrence id.

    Idempotent: if this exact raw content has already been turned into an import by this exact set of decisions, nothing is
    re-read, re-archived, or re-saved.

    - `project_path` -- project to ingest into; created lazily if absent.
    - `import_csv_path` -- source CSV, read to build the occurrence table
      unless `df` is given. Also what gets archived as the raw import, byte
      for byte, unless `raw_bytes` says otherwise -- so a caller passing both
      `df` and `raw_bytes` needs this only as a name to log and to derive
      `name_prefix`/extension from; it need not exist on disk.
    - `id_col` -- source column identifying an occurrence. Omit if already
      called `occurrence_id`. Duplicates or blanks raise.
    - `image_url_col` -- source column holding the image URL. Optional --
      omit it when the images are local.
    - `datetime_cols`, `numeric_cols` -- source columns to coerce. Naming an
      absent column is harmless.
    - `transform` -- optional `callable(df) -> df` applied after
      normalization, for deriving columns a project needs. Recorded in the
      import manifest by name (see `_recorded_callable`) -- give it a real
      name rather than a lambda if what it does is worth being able to tell
      apart later.
    - `drop` -- `{column: values}` naming rows the source has already said
      are NOT organisms, e.g. `{"determination_name": ["Not Lepidoptera"]}`.
      This is the occurrence contract, not a filter: a row the source
      classified as debris asserts no organism, so it doesn't belong in a
      table whose every row does. Not for quality -- "blurred" or "score
      below 0.5" are judgements you will want to revise, so use export
      filters, which keep the data. Missing values never match; an unknown
      column raises; dropped rows stay in the archived raw import.
    - `group_col`, `max_per_group`, `cap_rule` -- cap ingest at
      `max_per_group` rows per distinct value of `group_col`, e.g.
      `group_col="species"`, `max_per_group=200` against a pull dominated by
      a few common species. This is orthogonal to `drop=`: `drop` says a row
      isn't an organism, this says a project has enough organisms of this
      kind already. Applied after `drop=`, so a row `drop=` excludes never
      counts toward its group's cap. Must be given together; see
      `selectionhelpers.cap_per_group` for `cap_rule`'s options and how a
      missing `group_col` value is handled. Occurrences already in the
      project are prioritized to stay, up to the cap, ahead of anything new
      this import adds -- so raising `max_per_group` on a later reimport of a
      growing pull is additive (same specimens, plus more), never a
      reshuffle that orphans work already done on specimens that are still
      perfectly good candidates. Lowering it re-applies `cap_rule` among
      what's already kept. A project's first ingest has nothing to
      prioritize, so it behaves exactly as if this sentence didn't exist.
    - `name_prefix` -- prefix for the archived raw import's filename.
    - `raw_bytes`, `raw_extension` -- archive THIS content as the raw import
      instead of `import_csv_path`'s own bytes, for a caller that already did
      some reshaping (e.g. merging two source tables into one row per
      occurrence) before handing this function a ready-to-parse CSV -- the
      true raw import is what came before that reshaping, and that reshaping
      itself belongs in `manifest_extra`. Extension-specific; core callers
      leave both at their defaults.
    - `manifest_extra` -- optional dict of additional decisions to fold into
      the import manifest and its hash, for a caller with its own structural
      or judgement calls upstream of this function (e.g.
      `gbif_darwincore_inat`'s `media_rule`, which photo of several
      represents an occurrence). Values should be JSON-serializable; a
      callable is recorded by name via `_recorded_callable`.
    - `df` -- use this table instead of reading `import_csv_path`, for a
      caller that already built it in memory (e.g. after merging two source
      tables) and would otherwise have to write it to a CSV purely to hand it
      to this function, then have this function read that CSV straight back
      -- a full extra write and re-parse of the whole table for no reason,
      and for a wide, `dtype=str` source, a second trip through pandas'
      type-inference on every column that isn't explicitly coerced below,
      silently undoing whatever the caller was careful to keep as a string.
    - `trust_source_file_unchanged` -- skip reading and hashing
      `import_csv_path`'s content when its path, size, and modification time
      exactly match a previous import already on record for this project,
      and reuse that import's `raw_hash` instead. For a multi-gigabyte
      source, this is the difference between a repeat, no-op ingest
      finishing instantly and it re-reading and re-hashing every byte just
      to discover nothing changed. This is NOT a content guarantee:
      path+size+mtime matching does not prove the bytes are unchanged. A
      file regenerated with coincidentally the same size, a
      touched-but-otherwise-identical file, clock skew, or a different file
      copied over the same path with the same size, would all be (wrongly)
      trusted. Off by default for exactly that reason -- turn it on only for
      a source you know is written atomically and never silently replaced in
      place. Every OTHER decision (`id_col`, `drop=`, `group_col=`, etc.) is
      still checked exactly as always, so changing one of those against an
      unchanged file still triggers a real re-ingest rather than being
      skipped.
    - `fingerprint_source_path` -- the file `trust_source_file_unchanged`'s
      fingerprint is actually about, when that isn't `import_csv_path` -- for
      a caller (e.g. `gbif_darwincore_inat`) that hands this function a
      synthetic name plus `raw_bytes=` it already read from a real file
      elsewhere. Recorded in the manifest and compared against on a later
      call; defaults to `import_csv_path`.

    Returns the resulting occurrence table.
    """
    if bool(group_col) != bool(max_per_group):
        raise ValueError(
            "group_col and max_per_group must be given together"
        )

    logger.info("ingest_occurrences: starting on %s -> %s", import_csv_path, project_path)

    import_csv_path = Path(import_csv_path)

    if trust_source_file_unchanged and raw_bytes is None:
        if fingerprint_unchanged(project_path, import_csv_path, id_col=id_col,
                image_url_col=image_url_col, datetime_cols=datetime_cols,
                numeric_cols=numeric_cols, transform=transform, drop=drop,
                group_col=group_col, max_per_group=max_per_group,
                cap_rule=cap_rule, manifest_extra=manifest_extra):
            logger.info("trusting %s is unchanged (same path, size, and "
                        "mtime as a previous import) -- skipping the content "
                        "hash entirely", import_csv_path)
            return occurrence_records.load_occurrences(project_path)

    if raw_bytes is None:
        logger.info("reading raw import bytes from %s", import_csv_path)
        raw_bytes = import_csv_path.read_bytes()
        raw_extension = raw_extension or import_csv_path.suffix

    logger.info("hashing raw import content (%.1f MB)", len(raw_bytes) / 1e6)
    start = time.monotonic()
    raw_hash = _content_hash(raw_bytes)
    logger.info("hashed in %.1fs", time.monotonic() - start)

    existing_imports = load_imports(project_path)
    known_raw_hashes = (
        dict(zip(existing_imports["raw_hash"], existing_imports["raw_path"]))
        if not existing_imports.empty else {})

    recipe = _import_recipe(raw_hash, id_col, image_url_col, datetime_cols,
                            numeric_cols, transform, drop, group_col,
                            max_per_group, cap_rule, manifest_extra)
    import_hash = hash_spec(recipe)

    if not existing_imports.empty and (existing_imports["import_hash"] == import_hash).any():
        logger.info("this raw import has already been ingested with these same "
                    "decisions (import_hash=%s) -- nothing to do", import_hash)
        return occurrence_records.load_occurrences(project_path)

    logger.info("archiving raw import (%.1f MB) to %s", len(raw_bytes) / 1e6,
               paths.raw_imports_dir(project_path))
    start = time.monotonic()
    raw_path, is_new = _archive_raw_import(project_path, raw_bytes, name_prefix,
                                           raw_extension or ".csv", known_raw_hashes)
    logger.info("archived in %.1fs", time.monotonic() - start)

    if df is not None:
        n_read = len(df)
        logger.info("using the %d-row table already in memory instead of "
                    "reading %s", n_read, import_csv_path)
    else:
        logger.info("reading occurrence table from %s", import_csv_path)
        start = time.monotonic()
        df = pd.read_csv(import_csv_path)
        n_read = len(df)
        logger.info("read %d rows from %s in %.1fs", n_read, import_csv_path,
                   time.monotonic() - start)

    logger.info("normalizing %d row(s) (id_col=%r, image_url_col=%r, "
               "%d datetime col(s), %d numeric col(s))", n_read, id_col,
               image_url_col, len(datetime_cols), len(numeric_cols))
    start = time.monotonic()
    df = occurrence_records.normalize(df, id_col=id_col,
                                      image_col=image_url_col,
                                      datetime_cols=datetime_cols,
                                      numeric_cols=numeric_cols)
    logger.info("normalized in %.1fs", time.monotonic() - start)

    if transform is not None:
        logger.info("applying transform %s to %d row(s)",
                   _recorded_callable(transform), len(df))
        start = time.monotonic()
        df = transform(df)
        logger.info("transformed in %.1fs (%d row(s) now)",
                   time.monotonic() - start, len(df))

    n_dropped = 0
    if drop:
        logger.info("checking %d row(s) against drop=%s", len(df), drop)
        start = time.monotonic()
        excluded = selectionhelpers.rows_matching(df, drop)
        n_dropped = int(excluded.sum())
        logger.info("dropped %d of %d row(s) the source declared are not "
                    "organisms (%s) in %.1fs; the archived raw import keeps "
                    "all of them", n_dropped, len(df), drop, time.monotonic() - start)
        df = df[~excluded].reset_index(drop=True)

    n_capped = 0
    if group_col:
        logger.info("capping %d row(s) to at most %d per '%s' (cap_rule=%r)",
                   len(df), max_per_group, group_col, cap_rule)
        start = time.monotonic()
        existing_ids = occurrence_records.load_occurrences(
            project_path, columns=[], missing_ok=True)[occurrence_records.ID_COL]
        before = len(df)
        df = selectionhelpers.cap_per_group(df, group_col, max_per_group,
                                            rule=cap_rule,
                                            id_col=occurrence_records.ID_COL,
                                            keep_ids=set(existing_ids))
        n_capped = before - len(df)
        logger.info("capped %d row(s) in %.1fs (%d remain)", n_capped,
                   time.monotonic() - start, len(df))

    logger.info("writing occurrence table (%d rows) to %s", len(df),
               paths.occurrences_path(project_path))
    table = occurrence_records.save_occurrences(project_path, df)

    fingerprint_path = (Path(fingerprint_source_path)
                       if fingerprint_source_path is not None else import_csv_path)
    try:
        _stat = fingerprint_path.stat()
        _source_bytes, _source_mtime = _stat.st_size, _stat.st_mtime
    except OSError:
        _source_bytes = _source_mtime = None

    record = dict(recipe, import_hash=import_hash, raw_path=str(raw_path),
                 raw_import_reused=not is_new,
                 import_source_path=str(fingerprint_path),
                 import_source_bytes=_source_bytes,
                 import_source_mtime=_source_mtime,
                 row_counts={"read": n_read, "dropped": n_dropped,
                            "capped": n_capped, "final": len(df)},
                 created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 project=str(paths.project_dir(project_path)))
    _write_import_manifest(project_path, record, raw_path, import_hash)

    return table


def _write_import_manifest(project_path, record, raw_path, import_hash):
    """
    Put the record beside the raw import and into the project's imports log.

    Beside it so raw_imports/ says what each raw import became without
    consulting the log; in the log so every import this project has ever
    produced can be listed without a directory walk.
    """
    sidecar = paths.import_sidecar_path(raw_path, import_hash)
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    logger.info("wrote import manifest -> %s", sidecar)

    log = paths.imports_log_path(project_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
    return record


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
    raw_imports/ would double a project's largest storage cost. There is no
    import manifest recording decisions the way ingest_occurrences writes one:
    a folder of images has no id_col or drop= to record, and this is not
    idempotent by design -- see the note above about re-reading every file.

    - `project_path` -- project to ingest into; created lazily.
    - `image_dir` -- directory of images.
    - `patterns` -- glob patterns to match.
    - `id_from_stem` -- derive each occurrence id from the filename stem. The
      only supported scheme, so filenames must be unique and stable --
      renaming one orphans every mask and metric keyed to it.
    - `metadata` -- optional DataFrame of extra occurrence columns joined on
      `occurrence_id`, whose values match the filename stems. How a metadata
      CSV and a folder of images become one project.
    - `recursive` -- search subdirectories too.
    - `batch_size` -- images written to the store per LMDB transaction.

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
    """Write the manifest of an image ingest into the project's raw_imports directory."""
    manifest_dir = paths.raw_imports_dir(project_path)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    temporary = manifest_dir / ".manifest.csv"
    df.to_csv(temporary, index=False)
    _archive_source_file(temporary, project_path, f"images_{directory.name}")
    temporary.unlink()
