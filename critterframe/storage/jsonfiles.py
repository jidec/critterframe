"""The JSON and JSON Lines files a project keeps: atomic_write, write_json, read_json, append_jsonl, read_jsonl."""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..core.recipes import json_default

logger = logging.getLogger(__name__)


@contextmanager
def atomic_write(path, mode="w"):
    """Open a file handle whose content replaces `path` only once writing finishes.

    Args:
        path: Destination; its parent directory is created if missing.
        mode: `"w"` for UTF-8 text, or `"wb"`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        encoding = None if "b" in mode else "utf-8"
        with open(tmp_path, mode, encoding=encoding) as handle:
            yield handle
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json(path, record, indent=2):
    """Write one record as sorted UTF-8 JSON, atomically, and return the path.

    Args:
        path: Destination file.
        record: JSON-serializable value; NumPy scalars and arrays are converted.
        indent: Indentation; None writes one line.
    """
    with atomic_write(path) as handle:
        json.dump(record, handle, indent=indent, sort_keys=True, default=json_default)
    return Path(path)


def read_json(path, default=None):
    """Read one JSON file, or return `default` if it isn't there.

    Args:
        path: File to read.
        default: Value returned when the file is missing.
    """
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path, record):
    """Append one record to a JSON Lines log, and return the path.

    Args:
        path: Log file; its parent directory is created if missing.
        record: JSON-serializable value, written as one line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        # A previous append killed partway through leaves a line with no
        # newline on it. Appending straight onto that would splice this record
        # into the broken one and lose both, rather than only the broken one.
        if handle.tell() and not _ends_with_newline(path):
            handle.write("\n")
        handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")
    return path


def _ends_with_newline(path):
    """Return whether a file's last byte is a newline, i.e. its last line is complete."""
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def read_jsonl(path, what="record"):
    """Read a JSON Lines log as a DataFrame, oldest first.

    A malformed line is skipped with a warning.

    Args:
        path: Log file.
        what: What one line is, for the warning, e.g. `"import"`.

    Returns:
        One row per record; empty if the file isn't there.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping unreadable %s on line %d of %s", what, number, path)
    return pd.DataFrame(records)
