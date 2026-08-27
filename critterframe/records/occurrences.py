"""
Occurrences: one focal organism, at a place and time, represented by one
analysis image with metadata - one row each.

One organism per image is the most important structural decision in
CritterFrame. Everything downstream leans on it: a mask belongs to an
occurrence-part, a metric belongs to an occurrence-part, and an export has one
row per occurrence.

If your images contain several organisms, they must be
detected and separated upstream of ingest; there is no supported way to carry
two focal organisms in one occurrence.

Parts are the thing that IS allowed to be plural. A dragonfly image with four
wings is one occurrence with four parts, not four occurrences.

The table is stored in project_path/occurrences.parquet. Beyond the two
canonical columns below, whatever the source provided is kept as-is:
normalization means giving the identity and image columns known names, not
throwing away everything else.

The table is always written as a SNAPSHOT -- there is no incremental add. An
ingest states what the project's occurrences are, in full, and the newest
statement wins. That keeps one obvious question ("what is in this project?")
answerable by looking at one source, rather than at an accumulated history of
partial imports whose order mattered. To add data, extend the source and
re-ingest it.
"""

import logging

import pandas as pd

from ..project import paths
from ..recipes import hash_spec
from ..storage.tables import load_table, write_table

logger = logging.getLogger(__name__)

# occurrence_id is the one column every project has, whatever it was ingested
# from -- everything downstream is keyed by it. image_url is OPTIONAL: it's how
# download_images() finds an image, and a project whose images came from a
# local folder has no URLs and doesn't need the column at all.
ID_COL = "occurrence_id"
IMAGE_URL_COL = "image_url"


def normalize(df, id_col=None, image_col=None, datetime_cols=(), numeric_cols=()):
    """
    Rename a source table's identity/image columns to the canonical ones and
    make sure the ids are usable.

    Ids are cast to string on the way in and kept as strings everywhere
    afterward. Sources number their occurrences (Antenna detection ids,
    iNaturalist observation ids) and the same id read back from parquet, from
    an LMDB key, or from a sqlite column would otherwise compare unequal across
    those three depending on integer-vs-string typing -- which is the kind of
    bug that shows up as "the pipeline processed nothing" and takes an hour to
    find.

    Everything else the source provided is kept untouched -- normalization
    means agreeing on the two columns the package needs, not discarding the
    rest.

    df            -- source DataFrame.
    id_col        -- column holding the occurrence identifier. Omit if the
                    source already calls it occurrence_id.
    image_col     -- column holding the image URL. Omit if the source already
                    calls it image_url, or if there are no URLs because the
                    images are local -- nothing requires this column, and only
                    download_images() reads it.
    datetime_cols -- source column names to parse as datetimes.
    numeric_cols  -- source column names to parse as numeric. Naming a column
                    the source didn't include is harmless and skipped, so a
                    caller can list a superset -- which every caller does,
                    since external exports vary in which optional columns they
                    carry. Unparseable values become NaT/NaN rather than
                    raising: one malformed timestamp shouldn't cost you the
                    whole ingest.
    """
    df = df.copy()
    renames = {}
    if id_col and id_col != ID_COL:
        renames[id_col] = ID_COL
    if image_col and image_col != IMAGE_URL_COL:
        renames[image_col] = IMAGE_URL_COL
    if renames:
        df = df.rename(columns=renames)

    if ID_COL not in df.columns:
        raise KeyError(
            f"no '{ID_COL}' column after normalization -- pass id_col to say "
            f"which source column identifies an occurrence (columns: "
            f"{sorted(df.columns)})"
        )

    for column in datetime_cols:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df[ID_COL] = df[ID_COL].astype(str)
    validate_ids(df, source=str(id_col or ID_COL))

    return df.reset_index(drop=True)


def validate_ids(df, source=None):
    """
    Raise unless every row has an occurrence id and no id repeats.

    Both of these used to drop the offending rows with a warning, which was
    wrong. A duplicate id is not a row to discard -- it means the source
    disagrees with the package's central rule that one occurrence is one
    organism in one image, and silently keeping whichever copy happened to come
    first picks an arbitrary winner and loses the other, with nothing but a log
    line to say so. The same goes for a row with no id: it can't be segmented,
    measured, or exported, because everything downstream is keyed by id.

    Neither is recoverable here, because the fix is a judgement about the DATA
    -- are these two rows the same specimen photographed twice, or two
    specimens that were given the same number? So this reports the problem and
    stops, rather than guessing.

    df     -- table with an occurrence_id column of strings.
    source -- the source column name, for a message that points at what to fix.
    """
    if ID_COL not in df.columns:
        raise KeyError(f"no '{ID_COL}' column to validate")

    named = f"'{source}'" if source and source != ID_COL else f"'{ID_COL}'"

    # After astype(str) a null has become one of these spellings.
    missing = df[ID_COL].isin(["", "nan", "None", "<NA>", "NaT"])
    if missing.any():
        rows = list(df.index[missing][:5])
        raise ValueError(
            f"{int(missing.sum())} row(s) have no occurrence id in column "
            f"{named} (first at row {rows}) -- every occurrence needs one, "
            "since masks, metrics, and images are all keyed by it"
        )

    duplicated = df[ID_COL].duplicated(keep=False)
    if duplicated.any():
        offenders = sorted(df.loc[duplicated, ID_COL].unique())
        shown = ", ".join(offenders[:5])
        more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        raise ValueError(
            f"{len(offenders)} duplicate occurrence id(s) in column {named}: "
            f"{shown}{more} -- one occurrence is one organism in one image, so "
            "a repeated id means the source has two rows for the same one. Fix "
            "it in the source and re-ingest."
        )

    return df


def save_occurrences(project_path, df):
    """
    Replace the working occurrence table with a full snapshot.

    Replace rather than merge: external exports are full snapshots, not deltas,
    so the newest one IS the complete desired state. The archived import in
    project_path/imports/ is the recovery path if a replacement is ever wrong.

    Ids are validated here as well as in normalize(), because this is the one
    place every write passes through -- ingest_images() builds its rows from
    filenames without going near normalize(), and two files whose stems collide
    would otherwise reach the table unchallenged.
    """
    validate_ids(df)
    return write_table(df, paths.occurrences_path(project_path))


def load_occurrences(project_path, columns=None, missing_ok=False):
    """
    Read the occurrence table.

    columns    -- optional list of column names to read off disk; all if None.
                 occurrence_id is always included, since every caller keys on it.
    missing_ok -- True returns an empty frame when nothing has been ingested
                 yet, instead of raising.
    """
    if columns is not None:
        columns = list(dict.fromkeys([ID_COL] + list(columns)))

    df = load_table(paths.occurrences_path(project_path), columns=columns,
                    missing_ok=missing_ok)
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    return df


def occurrence_ids(project_path):
    """Every occurrence id in the project, as strings, in table order."""
    return load_occurrences(project_path, columns=[ID_COL])[ID_COL].tolist()


def ids_digest(occurrence_ids):
    """
    A short stable digest of a SET of occurrence ids.

    For recording WHICH occurrences something covered where listing them would
    be absurd -- the training data behind a registered model, the contents of an
    exported dataset. Order-independent and duplicate-independent, because the
    thing being identified is the set: the same 4,000 specimens listed in a
    different order are the same training data, and a record that said otherwise
    would report a difference nobody made.

    Not a substitute for the ids themselves. It answers "is this the same set as
    that one", never "which ones were they" -- keep the manifest for that.
    """
    return hash_spec(sorted({str(occurrence_id) for occurrence_id in occurrence_ids}))


def occurrence_count(project_path):
    """How many occurrences the project holds, or 0 if nothing is ingested yet."""
    return len(load_occurrences(project_path, columns=[ID_COL], missing_ok=True))
