"""The import record: what an import is hashed from, the log and manifest it is written to, and whether one is on record."""

import logging
from pathlib import Path

from ..core.recipes import hash_spec, recorded_callable, recorded_rules
from ..project import paths
from ..storage.jsonfiles import append_jsonl, read_jsonl, write_json

logger = logging.getLogger(__name__)


def import_recipe(
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

    The one place the decisions are assembled, so the ingest and `fingerprint_unchanged`
    cannot hash the same inputs differently.
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


def import_identity(raw_hash, **decisions):
    """Return `(recipe, import_hash)` for raw content turned into an import under these decisions.

    Args:
        raw_hash: `archive.content_hash` of the raw import.
        **decisions: Every decision argument `ingest_occurrences` was called with.
    """
    recipe = import_recipe(raw_hash, **decisions)
    return recipe, hash_spec(recipe)


def is_recorded(existing_imports, import_hash):
    """Return whether the import log holds this import hash.

    Args:
        existing_imports: The log, as `load_imports` returns it.
        import_hash: The hash to look for.
    """
    return not existing_imports.empty and bool((existing_imports["import_hash"] == import_hash).any())


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
    _recipe, import_hash = import_identity(borrowed_raw_hash, **decisions)
    return is_recorded(existing_imports, import_hash)


def load_imports(project_path):
    """Return every import the project has produced, oldest first, as a DataFrame."""
    return read_jsonl(paths.imports_log_path(project_path), what="import")


def write_import_manifest(project_path, record, raw_path, import_hash):
    """Write the record beside the raw import and append it to the project's imports log."""
    sidecar = write_json(paths.import_sidecar_path(raw_path, import_hash), record)
    logger.info("wrote import manifest -> %s", sidecar)

    append_jsonl(paths.imports_log_path(project_path), record)
    return record
