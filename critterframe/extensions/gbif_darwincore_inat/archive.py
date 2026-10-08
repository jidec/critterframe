"""Read occurrence.txt and multimedia.txt out of a GBIF Darwin Core Archive, zipped or extracted."""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

from ...timing import timed

logger = logging.getLogger(__name__)

OCCURRENCE_FILENAME = "occurrence.txt"
MULTIMEDIA_FILENAME = "multimedia.txt"

# The join key between the two tables -- always kept, whatever usecols a
# caller narrows either table to, since dropping it would silently break
# ingest.py's merge rather than raise something a caller could act on.
GBIF_ID_COL = "gbifID"


def _with_join_key(usecols):
    """Return `usecols` plus `GBIF_ID_COL`, or None when `usecols` is None."""
    if usecols is None:
        return None
    return list(dict.fromkeys([GBIF_ID_COL, *usecols]))


# GBIF's own field order for the *verbatim* Multimedia extension (what
# verbatim/multimedia.txt uses -- see the archive's meta.xml). Some GBIF
# downloads leak a row in exactly this shape into the processed multimedia.txt
# for records from one constituent dataset, instead of reshaping it to that
# table's own column order -- a GBIF export bug, not a project data problem.
# It's every core Multimedia column plus datasetKey and datasetID, so a row in
# this shape can always be remapped by name onto a core-shaped header.
VERBATIM_MULTIMEDIA_FIELDS = [
    "gbifID",
    "datasetKey",
    "type",
    "format",
    "identifier",
    "references",
    "title",
    "description",
    "created",
    "creator",
    "contributor",
    "publisher",
    "audience",
    "source",
    "license",
    "rightsHolder",
    "datasetID",
]


def read_darwincore_table(source, usecols=None):
    """Read one Darwin Core text table, tab-separated, with every column a string.

    Only an empty field reads as missing. A row with the wrong field count is recovered if
    it is a leaked verbatim Multimedia row, and otherwise dropped and logged.

    Args:
        source: A path, raw bytes, or a seekable file-like object.
        usecols: Column names to read; all if None. Worth narrowing for a large export,
            whose `occurrence.txt` has over 200 columns.
    """
    buffer = io.BytesIO(source) if isinstance(source, bytes) else source
    kwargs = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    try:
        return pd.read_csv(buffer, usecols=usecols, **kwargs)
    except pd.errors.ParserError:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        header = _peek_header(buffer)
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        dropped = []
        recovered = []

        def _recover_or_drop(bad_line):
            row = _recover_verbatim_multimedia_row(bad_line, header)
            if row is not None:
                recovered.append(bad_line)
                return row
            dropped.append(bad_line)
            return None

        df = pd.read_csv(buffer, engine="python", on_bad_lines=_recover_or_drop, **kwargs)
        if recovered:
            logger.warning(
                "recovered %d leaked verbatim Multimedia row(s) with the wrong field count", len(recovered)
            )
        if dropped:
            logger.warning(
                "dropped %d malformed row(s) with the wrong field count: %s", len(dropped), dropped
            )
        return df if usecols is None else df.loc[:, [c for c in usecols if c in df.columns]]


def _peek_header(buffer):
    """Return the header line of `buffer`, as a list of column names."""
    if isinstance(buffer, Path):
        with open(buffer, encoding="utf-8") as handle:
            line = handle.readline()
    else:
        position = buffer.tell()
        line = buffer.readline().decode("utf-8")
        buffer.seek(position)
    return line.rstrip("\r\n").split("\t")


def _recover_verbatim_multimedia_row(bad_line, header):
    """Return `bad_line` remapped onto the header's column order, or None.

    Recovers only a row with `VERBATIM_MULTIMEDIA_FIELDS`' field count under a header whose
    columns are all among those fields.
    """
    if len(bad_line) != len(VERBATIM_MULTIMEDIA_FIELDS):
        return None
    if not set(header) <= set(VERBATIM_MULTIMEDIA_FIELDS):
        return None
    by_term = dict(zip(VERBATIM_MULTIMEDIA_FIELDS, bad_line))
    return [by_term[column] for column in header]


def _find_one(names, filename):
    """Return the name matching `filename` case-insensitively, at the shallowest depth.

    A root-level table wins over a copy under `verbatim/`. Raises if several match at
    that depth.
    """
    matches = [name for name in names if Path(name).name.lower() == filename.lower()]
    if not matches:
        raise FileNotFoundError(f"no {filename} found (looked at: {sorted(names)})")
    shallowest = min(len(Path(name).parts) for name in matches)
    matches = [name for name in matches if len(Path(name).parts) == shallowest]
    if len(matches) > 1:
        raise ValueError(f"more than one {filename} found at the same depth: {matches}")
    return matches[0]


def read_darwincore_archive(path, occurrence_usecols=None, multimedia_usecols=None):
    """Read an archive's occurrence and multimedia tables.

    A zip member is streamed into the parser, never read whole into memory first.

    Args:
        path: A `.zip` as GBIF publishes it, or a directory it was extracted into. The two
            files are found by name, case-insensitively, at any depth.
        occurrence_usecols: Columns of `occurrence.txt` to read; all if None. `GBIF_ID_COL`
            is always kept.
        multimedia_usecols: Columns of `multimedia.txt` to read, likewise.

    Returns:
        `(occurrence_df, multimedia_df)`.
    """
    path = Path(path)
    occurrence_usecols = _with_join_key(occurrence_usecols)
    multimedia_usecols = _with_join_key(multimedia_usecols)
    logger.info("reading Darwin Core archive from %s", path)

    if path.is_dir():
        members = {str(candidate): candidate for candidate in path.rglob("*") if candidate.is_file()}
        occurrence_path = members[_find_one(members, OCCURRENCE_FILENAME)]
        multimedia_path = members[_find_one(members, MULTIMEDIA_FILENAME)]

        logger.info("parsing %s (%.1f MB)", occurrence_path, occurrence_path.stat().st_size / 1e6)
        with timed("parsed occurrence.txt", logger.info) as done:
            occurrence_df = read_darwincore_table(occurrence_path, usecols=occurrence_usecols)
            done.update(rows=len(occurrence_df), cols=len(occurrence_df.columns))

        logger.info("parsing %s (%.1f MB)", multimedia_path, multimedia_path.stat().st_size / 1e6)
        with timed("parsed multimedia.txt", logger.info) as done:
            multimedia_df = read_darwincore_table(multimedia_path, usecols=multimedia_usecols)
            done.update(rows=len(multimedia_df), cols=len(multimedia_df.columns))

    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            occurrence_name = _find_one(names, OCCURRENCE_FILENAME)
            multimedia_name = _find_one(names, MULTIMEDIA_FILENAME)

            info = archive.getinfo(occurrence_name)
            logger.info(
                "streaming and parsing %s (%.1f MB compressed, %.1f MB uncompressed)",
                occurrence_name,
                info.compress_size / 1e6,
                info.file_size / 1e6,
            )
            with timed("parsed occurrence.txt", logger.info) as done:
                with archive.open(occurrence_name) as stream:
                    occurrence_df = read_darwincore_table(stream, usecols=occurrence_usecols)
                done.update(rows=len(occurrence_df), cols=len(occurrence_df.columns))

            info = archive.getinfo(multimedia_name)
            logger.info(
                "streaming and parsing %s (%.1f MB compressed, %.1f MB uncompressed)",
                multimedia_name,
                info.compress_size / 1e6,
                info.file_size / 1e6,
            )
            with timed("parsed multimedia.txt", logger.info) as done:
                with archive.open(multimedia_name) as stream:
                    multimedia_df = read_darwincore_table(stream, usecols=multimedia_usecols)
                done.update(rows=len(multimedia_df), cols=len(multimedia_df.columns))

    else:
        raise ValueError(f"{path} is neither a directory nor a zip archive")

    logger.info(
        "read %d occurrence row(s) and %d multimedia row(s) from %s",
        len(occurrence_df),
        len(multimedia_df),
        path,
    )
    return occurrence_df, multimedia_df


def raw_archive_bytes(path):
    """Return the bytes to archive as this Darwin Core Archive's raw import.

    A `.zip` is returned as it is; a directory has its two tables zipped together
    untouched. Independent of which columns an ingest reads.

    Args:
        path: As `read_darwincore_archive` takes it.

    Returns:
        `(bytes, extension)`.
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        return path.read_bytes(), path.suffix or ".zip"

    if path.is_dir():
        members = {str(candidate): candidate for candidate in path.rglob("*") if candidate.is_file()}
        occurrence_path = members[_find_one(members, OCCURRENCE_FILENAME)]
        multimedia_path = members[_find_one(members, MULTIMEDIA_FILENAME)]

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(occurrence_path, arcname=OCCURRENCE_FILENAME)
            bundle.write(multimedia_path, arcname=MULTIMEDIA_FILENAME)
        return buffer.getvalue(), ".zip"

    raise ValueError(f"{path} is neither a directory nor a zip archive")
