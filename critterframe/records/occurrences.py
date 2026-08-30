"""
Normalize + save/load the occurrence table.

One organism per image is the structural decision everything downstream leans
on: a mask belongs to an occurrence-part, a metric belongs to an
occurrence-part, and an export has one row per occurrence. Images holding
several organisms must be separated upstream of ingest. Parts are the thing
that IS allowed to be plural -- a dragonfly with four wings is one occurrence
with four parts.

Ids are strings everywhere. Beyond the two canonical columns, whatever the
source provided is kept as-is: normalization means agreeing on two column
names, not discarding the rest.
"""

import logging

import pandas as pd

from ..project import paths
from ..recipes import hash_spec
from ..storage.tables import load_table, table_columns, write_table

logger = logging.getLogger(__name__)

# occurrence_id is the one column every project has, whatever it was ingested
# from -- everything downstream is keyed by it. image_url is OPTIONAL: it's how
# download_images() finds an image, and a project whose images came from a
# local folder has no URLs and doesn't need the column at all.
ID_COL = "occurrence_id"
IMAGE_URL_COL = "image_url"

def normalize(df, id_col=None, image_col=None, datetime_cols=(), numeric_cols=()):
    """
    Rename a source table's identity/image columns to the canonical ones and make
    sure the ids are usable.

    Ids are cast to string and stay strings everywhere after. Sources number
    their occurrences, and the same id read back from parquet, an LMDB key, and a
    sqlite column would otherwise compare unequal depending on typing -- a bug
    that shows up as "the pipeline processed nothing".

    Everything else the source provided is kept untouched.

    df            -- source DataFrame.
    id_col        -- column holding the occurrence identifier. Omit if already
                     called occurrence_id.
    image_col     -- column holding the image URL. Omit if already called
                     image_url, or if the images are local.
    datetime_cols -- source columns to parse as datetimes.
    numeric_cols  -- source columns to parse as numeric. Naming an absent column
                     is harmless, so a caller can list a superset. Unparseable
                     values become NaT/NaN rather than raising -- one malformed
                     timestamp shouldn't cost the whole ingest.
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

    Neither is recoverable automatically: a duplicate id means the source
    disagrees with the rule that one occurrence is one organism in one image, and
    keeping whichever copy came first picks an arbitrary winner and loses the
    other. A row with no id can't be segmented, measured, or exported. The fix is
    a judgement about the data -- one specimen photographed twice, or two
    specimens given one number? -- so this reports and stops.

    df     -- table with an occurrence_id column of strings.
    source -- the source column name, so the message points at what to fix.
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


def require_columns(project_path, columns, purpose):
    """
    Raise unless the occurrence table has every named column, listing the ones it
    does have.

    Checked against the parquet SCHEMA rather than by catching a read failure,
    which reports a pyarrow FieldRef error naming neither the column nor what
    wanted it. A mistyped column is nearly always a near-miss on one that is
    there, so listing them is usually the whole fix.

    columns -- one column name or an iterable of them. Empty is a no-op.
    purpose -- what the column would have been for, completing "...so there is
               <purpose>", e.g. "nothing to key a calibration on". Shared check,
               per-caller sentence.
    """
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns or [])
    if not columns:
        return columns

    available = table_columns(paths.occurrences_path(project_path))
    unknown = [column for column in columns if column not in available]
    if unknown:
        raise KeyError(
            f"this project's occurrences have no column(s) {unknown}, so there "
            f"is {purpose} (columns: {sorted(available)})"
        )
    return columns


def ids_digest(occurrence_ids):
    """
    A short stable digest of a SET of occurrence ids.

    For recording which occurrences something covered where listing them would be
    absurd: the training data behind a registered model, an exported dataset.
    Order- and duplicate-independent, since the same 4,000 specimens in a
    different order are the same training data.

    Answers "is this the same set", never "which ones were they" -- keep the
    manifest for that.
    """
    return hash_spec(sorted({str(occurrence_id) for occurrence_id in occurrence_ids}))


def occurrence_count(project_path):
    """How many occurrences the project holds, or 0 if nothing is ingested yet."""
    return len(load_occurrences(project_path, columns=[ID_COL], missing_ok=True))
