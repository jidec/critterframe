"""Ingest occurrence tables and local images, archiving the raw import and recording how it became occurrences."""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import drivers
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
    """Return a short, stable digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def _import_recipe(
    raw_hash,
    id_col=None,
    image_url_col=None,
    datetime_cols=(),
    numeric_cols=(),
    transform=None,
    drop=None,
    group_col=None,
    max_per_group=None,
    cap_rule="random",
    manifest_extra=None,
    dedupe_key_cols=None,
    dedupe_precision=None,
    dedupe_rule="random",
    prefer=None,
):
    """Return the dict `import_hash` is computed from.

    The one place the decisions are assembled, so the ingest, `fingerprint_unchanged` and
    `already_ingested` cannot hash the same inputs differently.
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
        **(
            {
                "dedupe": {
                    "key_cols": list(dedupe_key_cols),
                    "precision": dict(dedupe_precision or {}),
                    "rule": recorded_callable(dedupe_rule),
                }
            }
            if dedupe_key_cols
            else {}
        ),
        # Likewise recorded only when set.
        **({"prefer": recorded_rules(prefer)} if prefer else {}),
    }


def _transform_steps(transform):
    """Return `transform=` as a list of callables: one, several, or none."""
    if transform is None:
        return []
    if isinstance(transform, (list, tuple)):
        return list(transform)
    return [transform]


def _existing_ids(project_path):
    """Return the occurrence ids the project already holds."""
    table = occurrence_records.load_occurrences(project_path, columns=[], missing_ok=True)
    return set(table[occurrence_records.ID_COL])


def _fingerprint_raw_hash(existing_imports, source_path, stat):
    """Return the `raw_hash` of the newest import recorded for this source path, size and mtime, or None."""
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
    """Return whether a source file's path, size and mtime match an import already on record.

    Not a guarantee about its content: see `ingest_occurrences`' `trust_source_file_unchanged`.

    Args:
        project_path: Project whose import log to check.
        source_path: The file to match.
        **decisions: Every decision argument `ingest_occurrences` will be called with.
            They are hashed together, so a different value answers for a different import.
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
    """Return whether this raw content, under these decisions, has already been ingested.

    Args:
        project_path: Project whose import log to check.
        raw_bytes: The raw content that would be archived.
        **decisions: Every decision argument `ingest_occurrences` will be called with.
    """
    existing_imports = load_imports(project_path)
    if existing_imports.empty:
        return False
    raw_hash = _content_hash(raw_bytes)
    recipe = _import_recipe(raw_hash, **decisions)
    import_hash = hash_spec(recipe)
    return bool((existing_imports["import_hash"] == import_hash).any())


def load_imports(project_path):
    """Return every import the project has produced, oldest first, as a DataFrame."""
    return read_jsonl(paths.imports_log_path(project_path), what="import")


def _archive_raw_import(project_path, data, name_prefix, extension, known_raw_hashes):
    """Write raw content into `raw_imports/` as a new dated file, or reuse an identical one.

    Args:
        project_path: Project to archive into.
        data: The raw bytes.
        name_prefix: Prefix of the archived filename.
        extension: Its extension.
        known_raw_hashes: `{raw_hash: path string}` of the raw imports the log knows.

    Returns:
        `(path, is_new)`.
    """
    raw_hash = _content_hash(data)
    existing = _archived_raw_file(project_path, known_raw_hashes.get(raw_hash))
    if existing is not None:
        logger.info("raw import is byte-identical to %s -- reusing it", existing)
        return existing, False

    directory = paths.raw_imports_dir(project_path)
    directory.mkdir(parents=True, exist_ok=True)
    dest = paths.raw_import_path(project_path, name_prefix, extension=extension)
    dest.write_bytes(data)
    logger.info("archived raw import -> %s", dest)
    return dest, True


def _archived_raw_file(project_path, stored):
    """Return the archived raw file a log's `raw_path` names, if it is still there.

    An absolute path from an older log falls back to the same file name in this project's
    own `raw_imports/`.
    """
    if not isinstance(stored, str):
        return None
    candidates = [paths.resolve_in_project(project_path, stored)]
    if paths.is_absolute_anywhere(stored):
        candidates.append(paths.raw_imports_dir(project_path) / paths.file_name_anywhere(stored))
    return next((path for path in candidates if path.exists()), None)


def _archive_source_file(source_path, project_path, name_prefix):
    """Copy a source file into the project's raw-import archive, and return the archive path."""
    source_path = Path(source_path)
    return _archive_raw_import(
        project_path, source_path.read_bytes(), name_prefix, source_path.suffix, known_raw_hashes={}
    )[0]


def ingest_occurrences(
    project_path,
    import_csv_path,
    id_col=None,
    image_url_col=None,
    datetime_cols=(),
    numeric_cols=(),
    transform=None,
    drop=None,
    group_col=None,
    max_per_group=None,
    cap_rule="random",
    name_prefix="occurrences",
    raw_bytes=None,
    raw_extension=None,
    manifest_extra=None,
    df=None,
    trust_source_file_unchanged=False,
    fingerprint_source_path=None,
    visualize=True,
    stage_counts=None,
    dedupe_key_cols=None,
    dedupe_precision=None,
    dedupe_rule="random",
    prefer=None,
    read=None,
    raw=None,
):
    """Turn a raw import into the project's occurrence table, as a full snapshot.

    The raw import is archived before anything is parsed, and a manifest of every decision
    is written beside it. The same raw content under the same decisions is recognized and
    skipped.

    Args:
        project_path: Project to ingest into; created if absent.
        import_csv_path: Source CSV, read to build the table and archived as the raw
            import, unless `df`, `read`, `raw_bytes` or `raw` supply those.
        id_col: Source column identifying an occurrence; omit if it is `occurrence_id`.
            Duplicates and blanks raise.
        image_url_col: Source column holding the image URL; omit for local images.
        datetime_cols: Source columns to parse as datetimes.
        numeric_cols: Source columns to parse as numbers. Every column not named in these
            two stays a string, read with no type inference.
        transform: A `callable(df) -> df` applied after normalization, or a sequence of
            them. Each is recorded in the manifest by name, so pass a sequence, not one
            wrapper, to stack several.
        drop: `{column: values}` naming rows the source says are not organisms, e.g.
            `{"determination_name": ["Not Lepidoptera"]}`. Membership only; not for
            quality judgements, which belong in export filters.
        group_col: Occurrence column to cap by, with `max_per_group`.
        max_per_group: Most rows kept per value of `group_col`. Applied after `drop`, and
            occurrences already in the project are kept first, so raising it later only adds.
        cap_rule: Which rows of an oversized group survive; see `selectionhelpers.cap_per_group`.
        name_prefix: Prefix of the archived raw import's filename.
        raw_bytes: Content to archive as the raw import in place of the file's own bytes.
        raw_extension: Extension for `raw_bytes`.
        manifest_extra: Dict of a caller's own upstream decisions to record and hash.
        df: A table to use in place of reading `import_csv_path`.
        trust_source_file_unchanged: Skip reading and hashing the source when its path,
            size and modification time match an import on record. Faster for a very large
            file, but those three matching does not prove the bytes are unchanged.
        fingerprint_source_path: The file that check is about, when it isn't `import_csv_path`.
        visualize: Write pipeline figures of the rows kept at each stage. Only when an
            import happens.
        stage_counts: Ordered `{stage: rows}` from a caller's own reshaping, drawn first.
            Not hashed.
        dedupe_key_cols: Columns that together identify one real occurrence; rows sharing
            them are folded into one. Off if None. Applied after `drop` and before the cap.
        dedupe_precision: `{column: ndigits}` to round a numeric key column by.
        dedupe_rule: Which row of a duplicate group survives.
        prefer: `{column: values}` rule ranking rows within deduplication and the cap; it
            excludes nothing by itself, and ranks above keeping existing occurrences.
        read: A `callable(import_csv_path) -> df`, or `-> (df, stage_counts)`, replacing
            the CSV read. Run only once the import is known to be new. Not hashed.
        raw: A `callable(import_csv_path) -> (bytes, extension)` giving the raw import to
            archive and hash. Ignored with `raw_bytes`.

    Returns:
        The resulting occurrence table.
    """
    if bool(group_col) != bool(max_per_group):
        raise ValueError("group_col and max_per_group must be given together")

    logger.info("ingest_occurrences: starting on %s -> %s", import_csv_path, project_path)

    import_csv_path = Path(import_csv_path)

    decisions = {
        "id_col": id_col,
        "image_url_col": image_url_col,
        "datetime_cols": datetime_cols,
        "numeric_cols": numeric_cols,
        "transform": transform,
        "drop": drop,
        "group_col": group_col,
        "max_per_group": max_per_group,
        "cap_rule": cap_rule,
        "manifest_extra": manifest_extra,
        "dedupe_key_cols": dedupe_key_cols,
        "dedupe_precision": dedupe_precision,
        "dedupe_rule": dedupe_rule,
        "prefer": prefer,
    }

    fingerprint_path = (
        Path(fingerprint_source_path) if fingerprint_source_path is not None else import_csv_path
    )

    # A directory's size says nothing about its contents, so only a real file
    # can be trusted on its fingerprint.
    if trust_source_file_unchanged and raw_bytes is None and fingerprint_path.is_file():
        if fingerprint_unchanged(project_path, fingerprint_path, **decisions):
            logger.info(
                "trusting %s is unchanged (same path, size, and "
                "mtime as a previous import) -- skipping the content "
                "hash entirely",
                fingerprint_path,
            )
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
        if not existing_imports.empty
        else {}
    )

    recipe = _import_recipe(raw_hash, **decisions)
    import_hash = hash_spec(recipe)

    if not existing_imports.empty and (existing_imports["import_hash"] == import_hash).any():
        logger.info(
            "this raw import has already been ingested with these same "
            "decisions (import_hash=%s) -- nothing to do",
            import_hash,
        )
        return occurrence_records.load_occurrences(project_path)

    logger.info(
        "archiving raw import (%.1f MB) to %s", len(raw_bytes) / 1e6, paths.raw_imports_dir(project_path)
    )
    with timed("archived", logger.info):
        raw_path, is_new = _archive_raw_import(
            project_path, raw_bytes, name_prefix, raw_extension or ".csv", known_raw_hashes
        )

    stages = dict(stage_counts or {})
    if df is not None:
        n_read = len(df)
        logger.info("using the %d-row table already in memory instead of reading %s", n_read, import_csv_path)
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

    logger.info(
        "normalizing %d row(s) (id_col=%r, image_url_col=%r, %d datetime col(s), %d numeric col(s))",
        n_read,
        id_col,
        image_url_col,
        len(datetime_cols),
        len(numeric_cols),
    )
    with timed("normalized", logger.info):
        df = occurrence_records.normalize(
            df, id_col=id_col, image_col=image_url_col, datetime_cols=datetime_cols, numeric_cols=numeric_cols
        )

    stages["read"] = n_read

    for step in _transform_steps(transform):
        logger.info("applying transform %s to %d row(s)", recorded_callable(step), len(df))
        with timed("transformed", logger.info) as done:
            df = step(df)
            done["rows"] = len(df)
        stages["after transform"] = len(df)

    n_dropped = 0
    if drop:
        logger.info("checking %d row(s) against drop=%s", len(df), drop)
        with timed("dropped rows the source declared are not organisms", logger.info) as done:
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
            logger.info("deduplicating %d row(s) on %s", len(df), ", ".join(dedupe_key_cols))
            with timed("deduplicated", logger.info) as done:
                before = len(df)
                df = selectionhelpers.dedupe_by(
                    df,
                    dedupe_key_cols,
                    precision=dedupe_precision,
                    rule=dedupe_rule,
                    id_col=occurrence_records.ID_COL,
                    keep_ids=_existing_ids(project_path),
                    prefer=prefer,
                )
                n_deduped = before - len(df)
                done.update(removed=n_deduped, remaining=len(df))
            stages["after dedupe"] = len(df)

    n_capped = 0
    group_counts = None
    if group_col:
        logger.info(
            "capping %d row(s) to at most %d per '%s' (cap_rule=%r)",
            len(df),
            max_per_group,
            group_col,
            cap_rule,
        )
        with timed("capped", logger.info) as done:
            before = len(df)
            before_groups = df[group_col].value_counts()
            df = selectionhelpers.cap_per_group(
                df,
                group_col,
                max_per_group,
                rule=cap_rule,
                id_col=occurrence_records.ID_COL,
                keep_ids=_existing_ids(project_path),
                prefer=prefer,
            )
            n_capped = before - len(df)
            done.update(removed=n_capped, remaining=len(df))
        stages[f"after cap ({max_per_group}/{group_col})"] = len(df)
        after_groups = df[group_col].value_counts()
        group_counts = {
            str(group): {"before": int(count), "after": int(after_groups.get(group, 0))}
            for group, count in before_groups.head(30).items()
        }

    logger.info("writing occurrence table (%d rows) to %s", len(df), paths.occurrences_path(project_path))
    table = occurrence_records.save_occurrences(project_path, df)

    try:
        _stat = fingerprint_path.stat()
        _source_bytes, _source_mtime = _stat.st_size, _stat.st_mtime
    except OSError:
        _source_bytes = _source_mtime = None

    record = dict(
        recipe,
        import_hash=import_hash,
        raw_path=paths.relative_to_project(project_path, raw_path),
        raw_import_reused=not is_new,
        import_source_path=str(fingerprint_path),
        import_source_bytes=_source_bytes,
        import_source_mtime=_source_mtime,
        row_counts={
            "read": n_read,
            "dropped": n_dropped,
            "deduped": n_deduped,
            "capped": n_capped,
            "final": len(df),
        },
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        project=str(paths.project_dir(project_path)),
    )
    _write_import_manifest(project_path, record, raw_path, import_hash)

    stages["final"] = len(df)
    with pipeline_visualization.open_report(
        project_path, f"ingest__{name_prefix}", import_hash, visualize=visualize, identity=recipe
    ).begin([]) as report:
        if report:
            report.figure(
                "stages", figures.funnel(stages, title=f"{import_csv_path.name}: rows kept at each stage")
            )
            if group_counts:
                report.figure(
                    "groups",
                    figures.bar_chart(
                        group_counts,
                        stacked=False,
                        ylabel="rows",
                        title=f"largest '{group_col}' groups before and after the cap",
                    ),
                )

    return table


def _write_import_manifest(project_path, record, raw_path, import_hash):
    """Write the record beside the raw import and append it to the project's imports log."""
    sidecar = write_json(paths.import_sidecar_path(raw_path, import_hash), record)
    logger.info("wrote import manifest -> %s", sidecar)

    append_jsonl(paths.imports_log_path(project_path), record)
    return record


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
        The `drivers.Tally.summary` dict plus `occurrences`, the rows in the resulting table.
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
    """Write the manifest of an image ingest into the project's `raw_imports` directory."""
    manifest_dir = paths.raw_imports_dir(project_path)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    temporary = manifest_dir / ".manifest.csv"
    df.to_csv(temporary, index=False)
    _archive_source_file(temporary, project_path, f"images_{directory.name}")
    temporary.unlink()
