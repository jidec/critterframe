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
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import segments as segment_iteration
from . import selectionhelpers
from .project import paths
from .records import occurrences as occurrence_records
from .recipes import hash_spec, recorded_callable, recorded_rules
from .storage.imagestore import ImageStore
from .storage.jsonfiles import append_jsonl, read_jsonl, write_json
from .timing import timed
from .visualization import figures
from .visualization import pipeline as pipeline_visualization
from .visualization.panels import annotate

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEFAULT_BATCH_SIZE = 100

# Every column read as a string, only an empty field as missing: pandas' own
# inference turns an id "007" into 7 and a value "NA" into NaN, silently.
_CSV_READ = dict(dtype=str, keep_default_na=False, na_values=[""])

def _content_hash(data):
    """A short, stable digest of raw bytes -- how a raw import is recognized again."""
    return hashlib.sha256(data).hexdigest()[:16]

def _import_recipe(raw_hash, id_col=None, image_url_col=None, datetime_cols=(),
                   numeric_cols=(), transform=None, drop=None, group_col=None,
                   max_per_group=None, cap_rule="random", manifest_extra=None,
                   dedupe_key_cols=None, dedupe_precision=None, dedupe_rule="random",
                   prefer=None):
    """
    The dict import_hash is computed from. Shared between ingest_occurrences'
    own idempotency check, fingerprint_unchanged() and already_ingested(), so
    no two of them can land on a different hash for the same inputs -- which
    is why every decision is keyword-defaulted here and forwarded as one dict
    rather than spelled out again in each of their signatures.
    """
    return {
        "raw_hash": raw_hash,
        "id_col": id_col,
        "image_url_col": image_url_col,
        "datetime_cols": sorted(datetime_cols),
        "numeric_cols": sorted(numeric_cols),
        "transform": recorded_callable(transform),
        "drop": recorded_rules(drop),
        "group_col": group_col,
        "max_per_group": max_per_group,
        "cap_rule": recorded_callable(cap_rule),
        "extra": manifest_extra or {},
        # Recorded only when deduplication is actually on, so turning it on is
        # a different import while every import made before this stage existed
        # keeps the hash it was archived under.
        **({"dedupe": {"key_cols": list(dedupe_key_cols),
                       "precision": dict(dedupe_precision or {}),
                       "rule": recorded_callable(dedupe_rule)}}
           if dedupe_key_cols else {}),
        # Likewise recorded only when set.
        **({"prefer": recorded_rules(prefer)} if prefer else {}),
    }


def _transform_steps(transform):
    """
    `transform=` as a list of callables: one, several, or none.

    A sequence is how an extension stacks its own derivations on a caller's
    without wrapping both in a closure -- a closure records only the wrapper's
    name in the manifest, so two different callers' transforms would share an
    import hash (see ingest_occurrences' `transform`).
    """
    if transform is None:
        return []
    if isinstance(transform, (list, tuple)):
        return list(transform)
    return [transform]


def _existing_ids(project_path):
    """
    The occurrence ids this project already holds, for the import-time selections that
    prioritize keeping them (`selectionhelpers.cap_per_group`/`dedupe_by`'s keep_ids).
    """
    table = occurrence_records.load_occurrences(project_path, columns=[],
                                                missing_ok=True)
    return set(table[occurrence_records.ID_COL])


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


def fingerprint_unchanged(project_path, source_path, **decisions):
    """
    Whether source_path's path, size, and mtime match a previous import well
    enough to trust that import's own raw_hash instead of reading and
    hashing source_path again.

    NOT a content guarantee -- see ingest_occurrences' trust_source_file_unchanged
    for the risk this accepts.

    - `source_path` -- the file whose path, size, and mtime to match.
    - `decisions` -- every decision argument `ingest_occurrences` will be
      called with (`id_col`, `drop`, `group_col`, ...). They are hashed
      together, so a mismatched one here answers a question about a
      different import.

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
    recipe = _import_recipe(borrowed_raw_hash, **decisions)
    import_hash = hash_spec(recipe)
    return bool((existing_imports["import_hash"] == import_hash).any())


def already_ingested(project_path, raw_bytes, **decisions):
    """
    Whether this exact raw content, turned into an import by this exact set of
    decisions, has already been ingested -- computed from raw_bytes alone,
    with no parsed table required.

    For asking before building a table at all; a caller whose parse is the
    expensive step can instead hand it to ingest_occurrences as `read=`, which
    runs it only once this answers False.

    - `raw_bytes` -- the raw content that would be archived.
    - `decisions` -- every decision argument `ingest_occurrences` will be
      called with (typically alongside `df=`). They are hashed together, so a
      mismatched value here answers a question about a different import.

    Returns True if project_path's import log already has an import with
    this exact hash.
    """
    existing_imports = load_imports(project_path)
    if existing_imports.empty:
        return False
    raw_hash = _content_hash(raw_bytes)
    recipe = _import_recipe(raw_hash, **decisions)
    import_hash = hash_spec(recipe)
    return bool((existing_imports["import_hash"] == import_hash).any())


def load_imports(project_path):
    """
    Every import this project has produced from a raw import, oldest first.

    Reads the log ingest_occurrences appends to, the same pattern
    export.load_exports reads back. Returns a DataFrame; empty where this
    project has never ingested an occurrence table.
    """
    return read_jsonl(paths.imports_log_path(project_path), what="import")


def _archive_raw_import(project_path, data, name_prefix, extension, known_raw_hashes):
    """
    Write `data` into raw_imports/ as a new dated file, unless a raw import
    with identical content is already archived there -- in which case that
    file is reused rather than duplicated, since the same bytes are the same
    raw import no matter how many different decisions get made from them.

    - `known_raw_hashes` -- {raw_hash: path string} of every raw import this
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
                       trust_source_file_unchanged=False, fingerprint_source_path=None,
                       visualize=True, stage_counts=None, dedupe_key_cols=None,
                       dedupe_precision=None, dedupe_rule="random", prefer=None,
                       read=None, raw=None):
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
      absent column is harmless. A CSV is read with no type inference, so
      every column NOT named here stays a string (an id `007` stays `"007"`,
      a value `"NA"` stays `"NA"`; only an empty field is missing).
    - `transform` -- optional `callable(df) -> df` applied after
      normalization, for deriving columns a project needs, or a SEQUENCE of
      them applied in order. Each is recorded in the import manifest by name
      (see `recipes.recorded_callable`) -- give it a real name rather than a
      lambda if what it does is worth being able to tell apart later. A
      sequence is what an extension adding its own derivations on top of a
      caller's should pass: wrapping both in one closure records only the
      wrapper's name, so two different callers' transforms would share an
      import hash and the second ingest would be skipped.
    - `drop` -- `{column: values}` naming rows the source has already said
      are NOT organisms, e.g. `{"determination_name": ["Not Lepidoptera"]}`.
      This is the occurrence contract, not a filter: a row the source
      classified as debris asserts no organism, so it doesn't belong in a
      table whose every row does. Not for quality -- "blurred" or "score
      below 0.5" are judgements you will want to revise, so use export
      filters, which keep the data. Missing values never match; an unknown
      column raises; dropped rows stay in the archived raw import. Values
      compare by type, so a rule on a column not in `numeric_cols` needs
      string values.
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
      callable is recorded by name via `recipes.recorded_callable`.
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
      a caller that hands this function a synthetic name plus `raw_bytes=`
      it already read from a real file elsewhere. Recorded in the manifest and compared against on a later
      call; defaults to `import_csv_path`.
    - `visualize` -- True (default): pipeline figures of the rows kept at
      each stage (read, transform, drop=, cap, final), plus the largest
      groups before and after the cap when `group_col` is given. Written
      only when an import actually happens. False writes nothing.
    - `stage_counts` -- optional ordered `{stage: rows}` from a caller's own
      reshaping upstream of this function, drawn ahead of this function's
      stages. Descriptive only; not part of the import hash.
    - `dedupe_key_cols`, `dedupe_precision`, `dedupe_rule` -- OFF by default;
      fold together rows that are the same real occurrence published more
      than once, fingerprinted on these columns (see
      `selectionhelpers.dedupe_by`). Applied AFTER `drop=` and before the
      group cap: a row the source declared absent must not be able to win a
      duplicate group and take the real sighting down with it, and a
      per-group cap should count real specimens rather than a sighting
      inflated by however many aggregators published it. A key column absent
      from the table skips deduplication with a warning rather than failing
      the ingest, since narrowing columns without one is unexceptional.
    - `prefer` -- optional `{column: values}` rule, e.g.
      `{"institutionCode": ["iNaturalist"]}`, ranking rows within
      deduplication and the group cap; it never excludes a row by itself.
      Deduplication never removes a preferred row and removes any other row
      sharing its fingerprint; the cap fills a group from preferred rows
      first and only then from the rest. Ranks above the existing-occurrence
      priority, so a preferred newcomer displaces an already-kept row that
      isn't preferred. See `selectionhelpers.cap_per_group`/`dedupe_by`.
    - `read` -- optional `callable(import_csv_path) -> df`, or `-> (df,
      stage_counts)`, replacing the default CSV read for a source whose own
      parse is the expensive step. Runs only once the import is known to be
      new, so an unchanged source is never parsed twice. Not in the import
      hash: put whatever changes its output in `manifest_extra`.
    - `raw` -- optional `callable(import_csv_path) -> (bytes, extension)`
      giving the raw import to archive and hash, when that isn't the file's
      own bytes (e.g. an extracted archive directory). Ignored when
      `raw_bytes` is given.

    Returns the resulting occurrence table.
    """
    if bool(group_col) != bool(max_per_group):
        raise ValueError(
            "group_col and max_per_group must be given together"
        )

    logger.info("ingest_occurrences: starting on %s -> %s", import_csv_path, project_path)

    import_csv_path = Path(import_csv_path)

    decisions = {
        "id_col": id_col, "image_url_col": image_url_col,
        "datetime_cols": datetime_cols, "numeric_cols": numeric_cols,
        "transform": transform, "drop": drop, "group_col": group_col,
        "max_per_group": max_per_group, "cap_rule": cap_rule,
        "manifest_extra": manifest_extra, "dedupe_key_cols": dedupe_key_cols,
        "dedupe_precision": dedupe_precision, "dedupe_rule": dedupe_rule,
        "prefer": prefer,
    }

    fingerprint_path = (Path(fingerprint_source_path)
                       if fingerprint_source_path is not None else import_csv_path)

    # A directory's size says nothing about its contents, so only a real file
    # can be trusted on its fingerprint.
    if (trust_source_file_unchanged and raw_bytes is None
            and fingerprint_path.is_file()):
        if fingerprint_unchanged(project_path, fingerprint_path, **decisions):
            logger.info("trusting %s is unchanged (same path, size, and "
                        "mtime as a previous import) -- skipping the content "
                        "hash entirely", fingerprint_path)
            return occurrence_records.load_occurrences(project_path)

    if raw_bytes is None:
        logger.info("reading raw import bytes from %s", import_csv_path)
        with timed("read raw import", logger.info) as done:
            if raw is not None:
                raw_bytes, raw_extension = raw(import_csv_path)
            else:
                raw_bytes = import_csv_path.read_bytes()
                raw_extension = raw_extension or import_csv_path.suffix
            done["MB"] = round(len(raw_bytes) / 1e6, 1)

    logger.info("hashing raw import content (%.1f MB)", len(raw_bytes) / 1e6)
    with timed("hashed", logger.info):
        raw_hash = _content_hash(raw_bytes)

    existing_imports = load_imports(project_path)
    known_raw_hashes = (
        dict(zip(existing_imports["raw_hash"], existing_imports["raw_path"]))
        if not existing_imports.empty else {})

    recipe = _import_recipe(raw_hash, **decisions)
    import_hash = hash_spec(recipe)

    if not existing_imports.empty and (existing_imports["import_hash"] == import_hash).any():
        logger.info("this raw import has already been ingested with these same "
                    "decisions (import_hash=%s) -- nothing to do", import_hash)
        return occurrence_records.load_occurrences(project_path)

    logger.info("archiving raw import (%.1f MB) to %s", len(raw_bytes) / 1e6,
               paths.raw_imports_dir(project_path))
    with timed("archived", logger.info):
        raw_path, is_new = _archive_raw_import(
            project_path, raw_bytes, name_prefix, raw_extension or ".csv",
            known_raw_hashes)

    stages = dict(stage_counts or {})
    if df is not None:
        n_read = len(df)
        logger.info("using the %d-row table already in memory instead of "
                    "reading %s", n_read, import_csv_path)
    elif read is not None:
        logger.info("parsing %s with %s", import_csv_path, recorded_callable(read))
        result = read(import_csv_path)
        if isinstance(result, tuple):
            df, read_stages = result
            stages.update(read_stages)
        else:
            df = result
        n_read = len(df)
    else:
        logger.info("reading occurrence table from %s", import_csv_path)
        with timed(f"read {import_csv_path}", logger.info) as done:
            df = pd.read_csv(import_csv_path, **_CSV_READ)
            n_read = len(df)
            done["rows"] = n_read

    logger.info("normalizing %d row(s) (id_col=%r, image_url_col=%r, "
               "%d datetime col(s), %d numeric col(s))", n_read, id_col,
               image_url_col, len(datetime_cols), len(numeric_cols))
    with timed("normalized", logger.info):
        df = occurrence_records.normalize(df, id_col=id_col,
                                          image_col=image_url_col,
                                          datetime_cols=datetime_cols,
                                          numeric_cols=numeric_cols)

    stages["read"] = n_read

    for step in _transform_steps(transform):
        logger.info("applying transform %s to %d row(s)",
                   recorded_callable(step), len(df))
        with timed("transformed", logger.info) as done:
            df = step(df)
            done["rows"] = len(df)
        stages["after transform"] = len(df)

    n_dropped = 0
    if drop:
        logger.info("checking %d row(s) against drop=%s", len(df), drop)
        with timed("dropped rows the source declared are not organisms",
                   logger.info) as done:
            excluded = selectionhelpers.rows_matching(df, drop)
            n_dropped = int(excluded.sum())
            done.update(dropped=n_dropped, of=len(df), rule=drop)
        logger.info("the archived raw import keeps all %d of them", len(df))
        df = df[~excluded].reset_index(drop=True)
        stages["after drop="] = len(df)

    n_deduped = 0
    if dedupe_key_cols:
        missing = [column for column in dedupe_key_cols if column not in df.columns]
        if missing:
            logger.warning("skipping deduplication -- %s not in this table", missing)
        else:
            logger.info("deduplicating %d row(s) on %s", len(df),
                       ", ".join(dedupe_key_cols))
            with timed("deduplicated", logger.info) as done:
                before = len(df)
                df = selectionhelpers.dedupe_by(
                    df, dedupe_key_cols, precision=dedupe_precision,
                    rule=dedupe_rule, id_col=occurrence_records.ID_COL,
                    keep_ids=_existing_ids(project_path), prefer=prefer)
                n_deduped = before - len(df)
                done.update(removed=n_deduped, remaining=len(df))
            stages["after dedupe"] = len(df)

    n_capped = 0
    group_counts = None
    if group_col:
        logger.info("capping %d row(s) to at most %d per '%s' (cap_rule=%r)",
                   len(df), max_per_group, group_col, cap_rule)
        with timed("capped", logger.info) as done:
            before = len(df)
            before_groups = df[group_col].value_counts()
            df = selectionhelpers.cap_per_group(
                df, group_col, max_per_group, rule=cap_rule,
                id_col=occurrence_records.ID_COL,
                keep_ids=_existing_ids(project_path), prefer=prefer)
            n_capped = before - len(df)
            done.update(removed=n_capped, remaining=len(df))
        stages[f"after cap ({max_per_group}/{group_col})"] = len(df)
        after_groups = df[group_col].value_counts()
        group_counts = {str(group): {"before": int(count),
                                     "after": int(after_groups.get(group, 0))}
                        for group, count in before_groups.head(30).items()}

    logger.info("writing occurrence table (%d rows) to %s", len(df),
               paths.occurrences_path(project_path))
    table = occurrence_records.save_occurrences(project_path, df)

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
                            "deduped": n_deduped, "capped": n_capped,
                            "final": len(df)},
                 created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 project=str(paths.project_dir(project_path)))
    _write_import_manifest(project_path, record, raw_path, import_hash)

    stages["final"] = len(df)
    with pipeline_visualization.open_report(
            project_path, f"ingest__{name_prefix}", import_hash, visualize=visualize,
            identity=recipe).begin([]) as report:
        if report:
            report.figure("stages", figures.funnel(
                stages, title=f"{import_csv_path.name}: rows kept at each stage"))
            if group_counts:
                report.figure("groups", figures.bar_chart(
                    group_counts, stacked=False, ylabel="rows",
                    title=f"largest '{group_col}' groups before and after the cap"))

    return table


def _write_import_manifest(project_path, record, raw_path, import_hash):
    """
    Put the record beside the raw import and into the project's imports log.

    Beside it so raw_imports/ says what each raw import became without
    consulting the log; in the log so every import this project has ever
    produced can be listed without a directory walk.
    """
    sidecar = write_json(paths.import_sidecar_path(raw_path, import_hash), record)
    logger.info("wrote import manifest -> %s", sidecar)

    append_jsonl(paths.imports_log_path(project_path), record)
    return record


def ingest_images(project_path, image_dir, patterns=DEFAULT_IMAGE_PATTERNS,
                  id_from_stem=True, metadata=None, recursive=False,
                  batch_size=DEFAULT_BATCH_SIZE, visualize=True, visualize_every=None):
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
    - `visualize` -- True (default), an int, or ids: a pipeline grid of
      thumbnails of what was ingested, captioned with dimensions, with every
      unreadable file listed in its sidecar. False writes nothing.
    - `visualize_every` -- also write a thumbnail grid every N files,
      sampled from that stretch only.

    Returns a summary dict (see `segments.Tally.summary`) plus
    `occurrences`: rows in the resulting table.
    """
    if not id_from_stem:
        raise ValueError("id_from_stem=False isn't supported -- ids come from filenames")

    directory = Path(image_dir)
    glob = directory.rglob if recursive else directory.glob
    image_paths = sorted({path for pattern in patterns for path in glob(pattern)})

    _check_unique_stems(image_paths)

    stems = [path.stem for path in image_paths]
    identity = {"kind": "ingest_images", "image_dir": str(directory),
                "patterns": list(patterns), "recursive": recursive,
                "files": occurrence_records.ids_record(stems)}
    report = pipeline_visualization.open_report(
        project_path, f"ingest_images__{directory.name}", hash_spec(identity),
        visualize=visualize, visualize_every=visualize_every,
        identity=identity).begin(stems)

    tally = segment_iteration.Tally(attempted=len(image_paths))
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
                image = cv2.imdecode(np.frombuffer(data, np.uint8),
                                     cv2.IMREAD_UNCHANGED)
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
    logger.info("image ingest complete: attempted=%d saved=%d failed=%d",
                tally.attempted, tally.processed, tally.failed)
    return tally.summary(occurrences=len(table))


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
