"""The raw-import archive: raw bytes written into `raw_imports/` once, and found again."""

import hashlib
import logging

from ..project import paths

logger = logging.getLogger(__name__)


def content_hash(data):
    """Return a short, stable digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def archive_raw_import(project_path, data, name_prefix, extension, known_raw_hashes):
    """Write raw content into `raw_imports/` as a new dated file, or reuse an identical one.

    Args:
        project_path: Project to archive into.
        data: The raw bytes.
        name_prefix: Prefix of the archived filename.
        extension: Its extension.
        known_raw_hashes: `{raw_hash: path string}` of the raw imports already archived.

    Returns:
        `(path, is_new)`.
    """
    raw_hash = content_hash(data)
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


def archived_hashes(project_path, name_prefix):
    """Return `{raw_hash: path string}` for the archived files of one name prefix, read from disk.

    For an archive no log records. Every matching file is read and hashed.

    Args:
        project_path: Project to read.
        name_prefix: Prefix the files were archived under.
    """
    directory = paths.raw_imports_dir(project_path)
    if not directory.exists():
        return {}
    return {
        content_hash(path.read_bytes()): paths.relative_to_project(project_path, path)
        # the date `paths.raw_import_path` appends, so a longer prefix isn't swept in
        for path in sorted(directory.glob(f"{name_prefix}_[0-9][0-9][0-9][0-9]-*"))
        if path.is_file() and not path.name.endswith(paths.IMPORT_SIDECAR_SUFFIX)
    }


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
