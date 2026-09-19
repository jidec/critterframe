"""
The JSON and JSON Lines files a project keeps: write_json, append_jsonl, read_jsonl.

Manifests (an import's, an export's, a report's), the model registry, and the append-only logs
beside them. Mechanics only, like every other module here: what goes in a record belongs to
whoever writes it.

Whole-file writes go through `atomic_write`, for the reason `tables` gives about parquet -- a
process killed mid-write leaves the previous complete file rather than a truncated one. UTF-8
everywhere, explicitly: a species name or a note with an accent in it is ordinary, and Windows
writes cp1252 when asked for nothing.
"""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..recipes import json_default

logger = logging.getLogger(__name__)


@contextmanager
def atomic_write(path, mode="w"):
    """
    A file handle whose content replaces `path` only once writing finishes.

    Writes to a uniquely-named temp file beside the destination and
    `os.replace()`s it into place -- atomic on the same volume on POSIX and
    Windows alike -- so a crash, a full disk or a Ctrl-C leaves the previous
    complete file rather than half of the new one.

    - `path` -- destination; its parent directory is created if missing.
    - `mode` -- `"w"` (UTF-8 text, the default) or `"wb"`.
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
    """
    Write one record as JSON, atomically, sorted, in UTF-8, and return the path.

    - `path` -- destination file.
    - `record` -- JSON-serializable value. NumPy scalars and arrays are
      converted the way every other spec in the package converts them.
    - `indent` -- None writes it on one line.
    """
    with atomic_write(path) as handle:
        json.dump(record, handle, indent=indent, sort_keys=True,
                  default=json_default)
    return Path(path)


def read_json(path, default=None):
    """
    Read one JSON file back, or return `default` if it isn't there.

    - `path` -- file to read.
    - `default` -- what to return when the file is missing.
    """
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path, record):
    """
    Append one record to a JSON Lines log, and return the path.

    JSON Lines rather than one JSON document because these logs only ever
    grow, and a read-merge-rewrite of a growing file is what an appending
    writer should not be doing. Appending a single line is atomic enough for
    that: nothing rewrites what is already there.

    - `path` -- log file; its parent directory is created if missing.
    - `record` -- JSON-serializable value, written as one line.
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
    """Whether a file's last byte is a newline -- i.e. its last line is complete."""
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def read_jsonl(path, what="record"):
    """
    Read a JSON Lines log as a DataFrame, oldest first; empty if it isn't there.

    A malformed line is skipped with a warning rather than failing the read: a
    log is a record of what happened, and one truncated line (a process killed
    mid-append) shouldn't cost the history either side of it.

    - `path` -- log file.
    - `what` -- what one line is, for the warning ("import", "export", ...).
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
                logger.warning("skipping unreadable %s on line %d of %s",
                               what, number, path)
    return pd.DataFrame(records)
