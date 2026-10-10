"""The occurrence table: normalize, validate ids, save, load, and digest sets of ids."""

import logging

import pandas as pd

from ..project import paths
from ..core.recipes import hash_spec
from ..storage.tables import load_table, table_columns, write_table

logger = logging.getLogger(__name__)

# occurrence_id is the one column every project has, whatever it was ingested
# from -- everything downstream is keyed by it. image_url is OPTIONAL: it's how
# download_images() finds an image, and a project whose images came from a
# local folder has no URLs and doesn't need the column at all.
ID_COL = "occurrence_id"
IMAGE_URL_COL = "image_url"


def normalize(df, id_col=None, image_col=None, datetime_cols=(), numeric_cols=()):
    """Rename a source table's id and image columns to the canonical ones, with ids as strings.

    Every other column is kept as it is.

    Args:
        df: Source DataFrame.
        id_col: Column holding the occurrence id; omit if it is already `occurrence_id`.
        image_col: Column holding the image URL; omit if it is already `image_url`, or if
            the images are local.
        datetime_cols: Columns to parse as datetimes; unparseable values become NaT.
        numeric_cols: Columns to parse as numbers; unparseable values become NaN. Naming an
            absent column is harmless.
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
    """Raise unless every row has an occurrence id and no id repeats.

    Args:
        df: Table with an `occurrence_id` column of strings.
        source: The source's own name for the id column, for the error message.

    Raises:
        ValueError: If an id is missing or duplicated.
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
    """Replace the occurrence table with a full snapshot, after validating its ids.

    Args:
        project_path: Project to write to.
        df: The whole occurrence table.
    """
    validate_ids(df)
    return write_table(df, paths.occurrences_path(project_path))


def load_occurrences(project_path, columns=None, missing_ok=False):
    """Read the occurrence table.

    Args:
        project_path: Project to read from.
        columns: Column names to read; all if None. `occurrence_id` is always included.
        missing_ok: Return an empty frame when nothing has been ingested, instead of raising.
    """
    if columns is not None:
        columns = list(dict.fromkeys([ID_COL] + list(columns)))

    df = load_table(paths.occurrences_path(project_path), columns=columns, missing_ok=missing_ok)
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    return df


def occurrence_ids(project_path):
    """Return every occurrence id in the project, as strings, in table order."""
    return load_occurrences(project_path, columns=[ID_COL])[ID_COL].tolist()


def require_columns(project_path, columns, purpose):
    """Raise unless the occurrence table has every named column.

    Args:
        project_path: Project to check.
        columns: One column name or an iterable of them; empty checks nothing.
        purpose: What the column was wanted for, completing "...so there is <purpose>" in
            the error message.

    Raises:
        KeyError: If a column is missing; the message lists the columns that exist.
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
    """Return a short stable digest of a set of occurrence ids.

    Independent of order and of duplicates.

    Args:
        occurrence_ids: The ids.
    """
    return hash_spec(sorted({str(occurrence_id) for occurrence_id in occurrence_ids}))


def ids_record(occurrence_ids):
    """Return a set of occurrence ids as `{"count", "ids_hash"}`.

    Args:
        occurrence_ids: Iterable of ids, consumed once.
    """
    occurrence_ids = list(occurrence_ids)
    return {"count": len(occurrence_ids), "ids_hash": ids_digest(occurrence_ids)}


def occurrence_count(project_path):
    """Return how many occurrences the project holds, 0 if nothing is ingested."""
    return len(load_occurrences(project_path, columns=[ID_COL], missing_ok=True))
