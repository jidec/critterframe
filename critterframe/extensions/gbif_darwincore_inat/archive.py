"""
Read occurrence.txt and multimedia.txt out of a GBIF Darwin Core Archive.

GBIF hands out a Darwin Core Archive as one .zip, and the two tables ingest.py
needs are tab-separated text inside it. Unzipping it to disk first is a step a
caller shouldn't have to remember: a .zip's members are read straight out of
it, so a project can point at the download exactly as GBIF produced it, or at
a folder it's already been extracted into.
"""

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
    """usecols plus GBIF_ID_COL, unless usecols is None (meaning "everything")."""
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
    "gbifID", "datasetKey", "type", "format", "identifier", "references",
    "title", "description", "created", "creator", "contributor", "publisher",
    "audience", "source", "license", "rightsHolder", "datasetID",
]


def read_darwincore_table(source, usecols=None):
    """
    Read one Darwin Core text table as tab-separated text, every column kept as
    a string.

    Left as strings so a stray value that looks numeric (a catalogNumber like
    "007", a stateProvince abbreviated "NA") isn't silently coerced; ingest.py's
    id/datetime/numeric coercion happens afterward, by column name, not by
    pandas' guess. keep_default_na is off for the same reason -- GBIF text
    is full of words ("NA", "None") that are real values, not blanks, and only
    a genuinely empty field should read as missing.

    A GBIF export occasionally leaks a row from a constituent dataset's own
    column layout into a table it doesn't match -- the wrong field count for
    that table's header. A leaked verbatim Multimedia row is recovered by name
    (see VERBATIM_MULTIMEDIA_FIELDS); anything else with the wrong field count
    is dropped and logged, rather than failing the whole read.

    - `source` -- a path, raw bytes, or a seekable file-like object (e.g. a
      `zipfile.ZipExtFile` opened on an archive member) -- read straight
      from it rather than requiring the caller to materialize the whole
      member as bytes first.
    - `usecols` -- column names to keep; every other column is never
      materialized. A processed `occurrence.txt` carries 200+ columns as
      `dtype=str`, each cell its own Python object -- reading only the
      handful a project actually uses is the difference between this
      fitting in memory and not, for a multi-gigabyte export. None
      (default) reads every column, as before. Ignored on the rare
      malformed-row fallback path below, where the full row has to be seen
      to be recovered or reported; the result is narrowed afterward
      instead.
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

        df = pd.read_csv(buffer, engine="python", on_bad_lines=_recover_or_drop,
                         **kwargs)
        if recovered:
            logger.warning("recovered %d leaked verbatim Multimedia row(s) with the "
                           "wrong field count", len(recovered))
        if dropped:
            logger.warning("dropped %d malformed row(s) with the wrong field count: %s",
                           len(dropped), dropped)
        return df if usecols is None else df.loc[:, [c for c in usecols if c in df.columns]]


def _peek_header(buffer):
    """The first (header) line of `buffer`, as a list of column names."""
    if isinstance(buffer, Path):
        with open(buffer, encoding="utf-8") as handle:
            line = handle.readline()
    else:
        position = buffer.tell()
        line = buffer.readline().decode("utf-8")
        buffer.seek(position)
    return line.rstrip("\r\n").split("\t")


def _recover_verbatim_multimedia_row(bad_line, header):
    """`bad_line` remapped onto `header`'s column order, or None if it isn't a
    leaked verbatim Multimedia row.

    Only fires when `bad_line` has exactly VERBATIM_MULTIMEDIA_FIELDS' field
    count and `header`'s columns are all among those fields -- true for a
    Multimedia table's header, essentially impossible for anything else, so
    this can't misfire on an unrelated malformed line in occurrence.txt.
    """
    if len(bad_line) != len(VERBATIM_MULTIMEDIA_FIELDS):
        return None
    if not set(header) <= set(VERBATIM_MULTIMEDIA_FIELDS):
        return None
    by_term = dict(zip(VERBATIM_MULTIMEDIA_FIELDS, bad_line))
    return [by_term[column] for column in header]


def _find_one(names, filename):
    """The name in `names` matching `filename` case-insensitively, at the shallowest depth.

    GBIF archives sometimes carry a second copy under `verbatim/` alongside the
    root-level table; when that happens the root-level file wins rather than
    raising, since it's the processed table `read_darwincore_table` expects.
    Still raises if more than one match sits at that same shallowest depth.
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
    """
    Read an archive's occurrence and multimedia tables.

    A multi-gigabyte GBIF export can take minutes just to decompress and parse
    -- this logs before and after each of the two slow steps (finding, then
    streaming and parsing each member) precisely because a caller watching the
    log otherwise sees nothing between "start" and "done" and has no way to
    tell a multi-minute parse from a hang.

    A .zip member is streamed straight into the parser rather than read into
    a bytes object first: zipfile decompresses a ZipExtFile on demand as it's
    read, so pandas' own internal read buffering is the only copy of the
    decompressed text that ever exists at once. Reading the member eagerly
    first (archive.read()) would hold the WHOLE decompressed table as a single
    Python bytes object -- for a multi-gigabyte occurrence.txt, that's another
    several gigabytes on top of the parsed DataFrame, for no reason usecols=
    doesn't already avoid.

    - `path` -- a .zip file as GBIF publishes it, or a directory it's
      already been extracted into. Either way the two files are found by
      name, case-insensitively, however deep they sit.
    - `occurrence_usecols`, `multimedia_usecols` -- narrow either table to
      these columns; `GBIF_ID_COL` is always kept regardless, since
      `ingest.py`'s merge needs it. A processed `occurrence.txt` is 200+
      columns wide, every one an individually-allocated Python string under
      `dtype=str` -- for a multi-gigabyte export, this is the difference
      between fitting in memory and not. None (default) reads every
      column, as before.

    Returns (occurrence_df, multimedia_df), both read by read_darwincore_table.
    """
    path = Path(path)
    occurrence_usecols = _with_join_key(occurrence_usecols)
    multimedia_usecols = _with_join_key(multimedia_usecols)
    logger.info("reading Darwin Core archive from %s", path)

    if path.is_dir():
        members = {str(candidate): candidate for candidate in path.rglob("*")
                  if candidate.is_file()}
        occurrence_path = members[_find_one(members, OCCURRENCE_FILENAME)]
        multimedia_path = members[_find_one(members, MULTIMEDIA_FILENAME)]

        logger.info("parsing %s (%.1f MB)", occurrence_path,
                   occurrence_path.stat().st_size / 1e6)
        with timed("parsed occurrence.txt", logger.info) as done:
            occurrence_df = read_darwincore_table(occurrence_path,
                                                  usecols=occurrence_usecols)
            done.update(rows=len(occurrence_df), cols=len(occurrence_df.columns))

        logger.info("parsing %s (%.1f MB)", multimedia_path,
                   multimedia_path.stat().st_size / 1e6)
        with timed("parsed multimedia.txt", logger.info) as done:
            multimedia_df = read_darwincore_table(multimedia_path,
                                                  usecols=multimedia_usecols)
            done.update(rows=len(multimedia_df), cols=len(multimedia_df.columns))

    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            occurrence_name = _find_one(names, OCCURRENCE_FILENAME)
            multimedia_name = _find_one(names, MULTIMEDIA_FILENAME)

            info = archive.getinfo(occurrence_name)
            logger.info("streaming and parsing %s (%.1f MB compressed, %.1f MB uncompressed)",
                       occurrence_name, info.compress_size / 1e6, info.file_size / 1e6)
            with timed("parsed occurrence.txt", logger.info) as done:
                with archive.open(occurrence_name) as stream:
                    occurrence_df = read_darwincore_table(
                        stream, usecols=occurrence_usecols)
                done.update(rows=len(occurrence_df),
                            cols=len(occurrence_df.columns))

            info = archive.getinfo(multimedia_name)
            logger.info("streaming and parsing %s (%.1f MB compressed, %.1f MB uncompressed)",
                       multimedia_name, info.compress_size / 1e6, info.file_size / 1e6)
            with timed("parsed multimedia.txt", logger.info) as done:
                with archive.open(multimedia_name) as stream:
                    multimedia_df = read_darwincore_table(
                        stream, usecols=multimedia_usecols)
                done.update(rows=len(multimedia_df),
                            cols=len(multimedia_df.columns))

    else:
        raise ValueError(f"{path} is neither a directory nor a zip archive")

    logger.info("read %d occurrence row(s) and %d multimedia row(s) from %s",
               len(occurrence_df), len(multimedia_df), path)
    return occurrence_df, multimedia_df


def raw_archive_bytes(path):
    """
    The bytes to archive as this Darwin Core Archive's raw import, without
    going through pandas at all.

    Byte-exact and independent of occurrence_usecols/multimedia_usecols on
    purpose: what's kept as the recovery copy must not depend on which columns
    a particular ingest happened to need. A .zip is archived verbatim; an
    already-extracted directory has its two Darwin Core tables zipped back
    together untouched, streamed straight from disk rather than read into a
    DataFrame and re-serialized.

    - `path` -- as read_darwincore_archive takes it.

    Returns (bytes, extension).
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        return path.read_bytes(), path.suffix or ".zip"

    if path.is_dir():
        members = {str(candidate): candidate for candidate in path.rglob("*")
                  if candidate.is_file()}
        occurrence_path = members[_find_one(members, OCCURRENCE_FILENAME)]
        multimedia_path = members[_find_one(members, MULTIMEDIA_FILENAME)]

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(occurrence_path, arcname=OCCURRENCE_FILENAME)
            bundle.write(multimedia_path, arcname=MULTIMEDIA_FILENAME)
        return buffer.getvalue(), ".zip"

    raise ValueError(f"{path} is neither a directory nor a zip archive")
