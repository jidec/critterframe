"""Ingest an occurrence table: archive the raw import, reshape and narrow it, and record how."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..project import paths
from ..core.recipes import recorded_callable
from ..records import occurrences as occurrence_records
from ..selection import algorithms as selection_algorithms
from ..timing import timed
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization
from .archive import archive_raw_import, content_hash
from .imports import fingerprint_unchanged, import_identity, is_recorded, load_imports, write_import_manifest

logger = logging.getLogger(__name__)

# Every column read as a string, only an empty field as missing: pandas' own
# inference turns an id "007" into 7 and a value "NA" into NaN, silently.
_CSV_READ = dict(dtype=str, keep_default_na=False, na_values=[""])


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
        cap_rule: Which rows of an oversized group survive; see `selection.algorithms.cap_per_group`.
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
        raw_hash = content_hash(raw_bytes)

    existing_imports = load_imports(project_path)
    known_raw_hashes = (
        dict(zip(existing_imports["raw_hash"], existing_imports["raw_path"]))
        if not existing_imports.empty
        else {}
    )

    recipe, import_hash = import_identity(raw_hash, **decisions)

    if is_recorded(existing_imports, import_hash):
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
        raw_path, is_new = archive_raw_import(
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
            excluded = selection_algorithms.rows_matching(df, drop)
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
                df = selection_algorithms.dedupe_by(
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
            df = selection_algorithms.cap_per_group(
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
    write_import_manifest(project_path, record, raw_path, import_hash)

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
