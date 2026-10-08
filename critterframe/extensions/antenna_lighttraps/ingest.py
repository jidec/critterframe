"""Ingest Antenna occurrence exports: the column mapping, the derived sheet and session columns, and what to drop."""

import logging
import os
import re
from urllib.parse import urlparse

import pandas as pd

from ...ingest import ingest_occurrences as core_ingest_occurrences
from ...project import paths
from . import api

logger = logging.getLogger(__name__)

ID_COL = "id"
URL_COL = "best_detection_url"

DATETIME_COLS = [
    "first_appearance_timestamp",
    "last_appearance_timestamp",
]

NUMERIC_COLS = [
    "determination_score",
    "detections_count",
    "best_detection_width",
    "best_detection_height",
]

# Determinations that mean "this detection isn't an organism", excluded at
# ingest (see ingest.ingest_occurrences' drop=). Values, not a confidence
# threshold, and the distinction is the point: this says Antenna decided the
# crop holds no moth, which is a fact about the source's output. How much to
# trust a LOW-CONFIDENCE identification of a real moth is a judgement about your
# own analysis -- keep those occurrences and filter them at export, where the
# threshold can be changed without re-ingesting.
NON_ORGANISM_DETERMINATIONS = {"determination_name": ["Not Lepidoptera"]}


def parse_sheet_image_id(url):
    """Return the sheet image a detection crop was cut from, parsed from its URL.

    For example `.../detections/199/2026-06-11/bronzeBobcat_2026_06_11__01_15_20_HDR0_detection_2121474.jpg`
    gives `bronzeBobcat/2026-06-11/bronzeBobcat_2026_06_11__01_15_20_HDR0.jpg`.

    Args:
        url: A `best_detection_url` value, or NaN.
    """
    if pd.isna(url):
        return pd.NA

    parts = urlparse(str(url)).path.strip("/").split("/")
    if len(parts) < 2:
        logger.warning("could not parse detection URL: %s", url)
        return pd.NA

    date_folder = parts[-2]
    stem, extension = os.path.splitext(parts[-1])

    # strip the occurrence-specific suffix, e.g. _detection_2121474
    sheet_stem = re.sub(r"_detection_[^/]+$", "", stem)
    if sheet_stem == stem:
        logger.warning("could not find a detection suffix in URL: %s", url)
        return pd.NA

    # the device name is the portion before the first underscore
    device = sheet_stem.split("_", 1)[0]
    return f"{device}/{date_folder}/{sheet_stem}{extension}"


def parse_session_path(sheet_image_id):
    """Return the device and date portion of a sheet image id, e.g. `bronzeBobcat/2026-06-11`.

    Args:
        sheet_image_id: A value from `parse_sheet_image_id`.
    """
    if pd.isna(sheet_image_id):
        return pd.NA

    parts = str(sheet_image_id).split("/")
    if len(parts) < 2:
        logger.warning("could not parse a session path from: %s", sheet_image_id)
        return pd.NA
    return "/".join(parts[:2])


def add_derived_columns(df):
    """Add `sheet_image_id` and `session_path`, derived from each crop's `image_url`."""
    from ...records.occurrences import IMAGE_URL_COL

    if IMAGE_URL_COL not in df.columns:
        raise KeyError(
            f"can't derive sheet_image_id: no '{IMAGE_URL_COL}' column (columns: {sorted(df.columns)})"
        )

    df = df.copy()
    df["sheet_image_id"] = df[IMAGE_URL_COL].map(parse_sheet_image_id)
    df["session_path"] = df["sheet_image_id"].map(parse_session_path)

    logger.info(
        "derived sheet_image_id for %d of %d occurrences, session_path for %d",
        int(df["sheet_image_id"].notna().sum()),
        len(df),
        int(df["session_path"].notna().sum()),
    )
    return df


def ingest_occurrences(
    project_path,
    import_csv_path=None,
    session=None,
    project=None,
    filters=None,
    transform=None,
    drop=NON_ORGANISM_DETERMINATIONS,
    group_col=None,
    max_per_group=None,
    cap_rule="random",
    visualize=True,
):
    """Ingest an Antenna occurrences export into a project, as a full snapshot.

    Args:
        project_path: Project to ingest into; created if absent.
        import_csv_path: A downloaded export CSV. None requests and downloads a fresh one,
            which needs credentials in the environment.
        session: Authenticated session to reuse.
        project: Antenna project id; from the environment if None.
        filters: Server-side export filters.
        transform: A `callable(df) -> df` run after the Antenna derivations.
        drop: Rows to exclude as non-organisms; `NON_ORGANISM_DETERMINATIONS` by default.
            None ingests every detection.
        group_col: As in `critterframe.ingest.ingest_occurrences`.
        max_per_group: As in `critterframe.ingest.ingest_occurrences`.
        cap_rule: As in `critterframe.ingest.ingest_occurrences`.
        visualize: As in `critterframe.ingest.ingest_occurrences`.

    Returns:
        The resulting occurrence table.
    """
    downloaded = None
    if import_csv_path is None:
        session = session or api.get_session()
        downloaded = paths.raw_imports_dir(project_path) / ".antenna_export.csv"
        import_csv_path = api.fetch_export(session, downloaded, project=project, filters=filters)

    # A sequence, not a closure around both: core records each transform by
    # name, so wrapping them would record only the wrapper's -- two different
    # callers' transforms would then share an import hash and the second
    # ingest would be skipped as already done.
    transforms = [add_derived_columns]
    if transform is not None:
        transforms.append(transform)

    try:
        return core_ingest_occurrences(
            project_path,
            import_csv_path,
            id_col=ID_COL,
            image_url_col=URL_COL,
            datetime_cols=DATETIME_COLS,
            numeric_cols=NUMERIC_COLS,
            transform=transforms,
            drop=drop,
            group_col=group_col,
            max_per_group=max_per_group,
            cap_rule=cap_rule,
            name_prefix=f"occurrences_antenna_{api.project_id(project)}",
            visualize=visualize,
        )
    finally:
        # The dated copy the core ingest archived is the durable record, so the
        # freshly downloaded temporary isn't needed once it's been ingested.
        if downloaded and downloaded.exists():
            downloaded.unlink()
