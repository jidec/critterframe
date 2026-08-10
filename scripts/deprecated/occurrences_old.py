"""
Design notes:
Every export is a full project snapshot, not a delta (for now).
We therefore refresh by replacing the working occurrences dataframe with the new one (for now).
Raw CSVs are kept dated and immutable under raw_exports.
"""

import os
import glob
import logging
from datetime import date

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

# The id column used only by load_with_masks()
JOIN_KEY = "id"

# Columns coerced to datetime / numeric on ingest. Extend to all columns later.
DATETIME_COLS = ["first_appearance_timestamp", "last_appearance_timestamp"]
NUMERIC_COLS = [
    "determination_score",
    "detections_count",
    "best_detection_width",
    "best_detection_height",
]

def exports_dir():
    return os.path.join(config.DATA_DIR, "raw_exports")

def occurrences_path():
    return os.path.join(config.DATA_DIR, "working_occurrences.parquet")

def masks_path():
    return os.path.join(config.DATA_DIR, "masks.parquet")

def copy_export_to_archive(csv_path):
    """
    Copy a freshly downloaded export CSV into the dated archive
    """
    out_dir = exports_dir()
    os.makedirs(out_dir, exist_ok=True)
    stamp = date.today().isoformat()
    base = f"occurrences_{config.PROJECT_ID}_{stamp}"

    dest = os.path.join(out_dir, f"{base}.csv")
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(out_dir, f"{base}_{n}.csv")
        n += 1

    with open(csv_path, "rb") as src, open(dest, "wb") as out:
        out.write(src.read())
    logger.info("Archived raw export -> %s", dest)
    return dest
 
def _typecast(df):
    """Type-cast known datetime/numeric columns."""
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def copy_export_to_working_occurrences(new_df):
    """
    Overwrite the working occurrences table (parquet) with the latest full
    snapshot. The newest snapshot is the complete desired state.
    """
    path = occurrences_path();
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = new_df.reset_index(drop=True)
    out.to_parquet(path)
    logger.info("Wrote working table -> %s (%d rows, full replace)", path, len(out))
    return out


def ingest_export(csv_path=os.path.join(config.DATA_DIR, "tmp", "antenna_export.csv")):
    """
    Ingest one downloaded export CSV:
      1. archive the raw file (dated, immutable)
      2. read + typecast
      3. replace the working parquet with this snapshot
    Returns the working dataframe.
    """
    copy_export_to_archive(csv_path)

    df = pd.read_csv(csv_path)
    logger.info("Read %d rows from %s", len(df), csv_path)
    df = _typecast(df)

    return copy_export_to_working_occurrences(df)


def load_occurrences(columns=None):
    """
    Read the working occurrence table for downstream steps.
    Pass columns=[...] to read only some columns off disk.
    Probably add more options later.
    """
    path = occurrences_path();
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No working table at {path}. Run an export ingest first."
        )
    return pd.read_parquet(path, columns=columns)


def load_with_masks(join_key=JOIN_KEY, how="inner"):
    """
    Join occurrences to their masks at read time.

    The two stores are kept separate (different write lifecycles) and this is
    where they come together. Default how='inner' returns only occurrences
    that have been segmented. Use how='left' to keep all occurrences with
    NaN mask columns where none exists yet.

    Assumes masks.parquet has a column matching `join_key` (the id).
    """
    occ = load_occurrences()
    mp = masks_path()
    if not os.path.exists(mp):
        raise FileNotFoundError(f"No mask store at {mp}. Run segmentation first.")
    masks = pd.read_parquet(mp)
    if join_key not in occ.columns:
        raise KeyError(
            f"Occurrences table has no '{join_key}' column to join on. "
            f"Columns: {list(occ.columns)}"
        )
    return occ.merge(masks, on=join_key, how=how)