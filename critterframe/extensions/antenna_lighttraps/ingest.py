"""
Normalize and ingest Antenna exports into a CritterFrame project.

Antenna occurrences map onto CritterFrame occurrences almost exactly, which is
what makes this extension short: Antenna has already detected individual
insects on a light-trap sheet and cut a crop per detection, so one Antenna
occurrence is one organism in one image -- the rule CritterFrame requires and
cannot check for you.

What this adds beyond a plain CSV ingest is the column mapping (Antenna's `id`
and `best_detection_url` become occurrence_id and image_url) and two derived
columns recovering where a crop came from: which sheet image it was cut out of,
and which deployment session that sheet belongs to. Both are lost otherwise --
the export doesn't carry them as fields, only encoded in the crop's URL -- and
a crop can then be traced back to the sheet it was cut from.

Classification exports are a separate registered format on the same API (see
api.CLASSIFICATIONS_FORMAT). They're not ingested here: a classification is a
determination OF an occurrence rather than an occurrence, so it belongs joined
onto the occurrence table as metadata columns. Fetch one with api.fetch_export
and merge it in via the `transform` hook below when that's wanted.

The occurrences export does carry one determination inline, though, and it
answers a different question: not WHAT the organism is but WHETHER there is one.
Antenna detects on a light-trap sheet before it classifies, so a detection can
turn out to be a leaf, a reflection, or a smear, and the export says so with
determination_name = "Not Lepidoptera". Those rows are dropped on the way in
(see NON_ORGANISM_DETERMINATIONS) -- a crop with no organism in it is not an
occurrence, and the occurrence table only means anything if every row in it is
one.
"""

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
    """
    Recover the source sheet image a detection crop was cut from, out of its
    URL.

    Example input:
        https://object-arbutus.cloud.computecanada.ca/ami-media-staging/
        uploads/detections/199/2026-06-11/
        bronzeBobcat_2026_06_11__01_15_20_HDR0_detection_2121474.jpg

    Example output:
        bronzeBobcat/2026-06-11/bronzeBobcat_2026_06_11__01_15_20_HDR0.jpg

    url -- a best_detection_url value (or NaN).
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
    """
    The device/date portion of a sheet_image_id -- the prefix every image from
    one deployment session shares, e.g.
    "bronzeBobcat/2026-06-11/bronzeBobcat_..._HDR0.jpg" ->
    "bronzeBobcat/2026-06-11".

    A reference card is set up once per session rather than once per image, so
    this is the key a scale measurement attaches to.
    """
    if pd.isna(sheet_image_id):
        return pd.NA

    parts = str(sheet_image_id).split("/")
    if len(parts) < 2:
        logger.warning("could not parse a session path from: %s", sheet_image_id)
        return pd.NA
    return "/".join(parts[:2])


def add_derived_columns(df):
    """
    Add sheet_image_id and session_path, derived from the crop URL.

    Runs after normalization, so it reads image_url rather than Antenna's
    original column name.
    """
    from ...records.occurrences import IMAGE_URL_COL

    if IMAGE_URL_COL not in df.columns:
        raise KeyError(
            f"can't derive sheet_image_id: no '{IMAGE_URL_COL}' column "
            f"(columns: {sorted(df.columns)})"
        )

    df = df.copy()
    df["sheet_image_id"] = df[IMAGE_URL_COL].map(parse_sheet_image_id)
    df["session_path"] = df["sheet_image_id"].map(parse_session_path)

    logger.info("derived sheet_image_id for %d of %d occurrences, "
                "session_path for %d",
                int(df["sheet_image_id"].notna().sum()), len(df),
                int(df["session_path"].notna().sum()))
    return df


def ingest_occurrences(project_path, import_csv_path=None, session=None,
                       project=None, filters=None, transform=None,
                       drop=NON_ORGANISM_DETERMINATIONS):
    """
    Ingest an Antenna occurrences export into a project, as a full snapshot.

    Snapshot is exactly right here rather than a simplification: every Antenna
    export IS a complete statement of the project's occurrences, not a delta,
    so the newest export is the newest truth. Re-ingesting on a schedule is the
    intended way to pick up new detections.

    project_path    -- project to ingest into; created lazily by the first
                      writer, so the directory needn't exist yet.
    import_csv_path -- an already-downloaded export CSV. Omit to request, wait
                      for, and download a fresh one via the API -- which needs
                      credentials in the environment (see api).
    session         -- authenticated session to reuse; one is created if
                      omitted and a download is needed.
    project         -- Antenna project id; from the environment if omitted.
    filters         -- optional server-side export filters.
    transform       -- optional callable(df) -> df run after the Antenna-specific
                      derivations, for anything project-specific (joining a
                      classification export, say).
    drop            -- rows to exclude as non-organisms, defaulting to
                      NON_ORGANISM_DETERMINATIONS. On by default because knowing
                      that "Not Lepidoptera" is Antenna's way of saying "no moth
                      here" is exactly the source-specific knowledge this
                      extension exists to hold; a project shouldn't have to
                      rediscover it. Pass None to ingest every detection --
                      which is what you want if you're auditing the classifier
                      itself rather than measuring moths.

    Returns the resulting occurrence table.
    """
    downloaded = None
    if import_csv_path is None:
        session = session or api.get_session()
        downloaded = paths.imports_dir(project_path) / ".antenna_export.csv"
        import_csv_path = api.fetch_export(session, downloaded, project=project,
                                           filters=filters)

    def antenna_transform(df):
        df = add_derived_columns(df)
        return transform(df) if transform is not None else df

    try:
        return core_ingest_occurrences(
            project_path,
            import_csv_path,
            id_col=ID_COL,
            image_url_col=URL_COL,
            datetime_cols=DATETIME_COLS,
            numeric_cols=NUMERIC_COLS,
            transform=antenna_transform,
            drop=drop,
            name_prefix=f"occurrences_antenna_{api.project_id(project)}",
        )
    finally:
        # The dated copy the core ingest archived is the durable record, so the
        # freshly downloaded temporary isn't needed once it's been ingested.
        if downloaded and downloaded.exists():
            downloaded.unlink()
