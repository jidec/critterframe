"""Ingest a GBIF Darwin Core Archive: one image per occurrence, merged and handed to the core ingest."""

import io
import logging
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd

from ... import ingest as core_ingest
from ...recipes import recorded_callable
from ...timing import timed
from . import archive

logger = logging.getLogger(__name__)

GBIF_ID_COL = "gbifID"
IDENTIFIER_COL = "identifier"
TYPE_COL = "type"
MEDIA_PREFIX = "media_"

DEFAULT_MEDIA_TYPE = "StillImage"

# GBIF's own way of saying "this record documents the absence of an organism",
# not a quality judgement -- see ingest_occurrences' drop= and the core
# contract it implements (critterframe.ingest.ingest_occurrences).
ABSENT_OCCURRENCES = {"occurrenceStatus": ["ABSENT"]}

# The rows iNaturalist itself published to GBIF, which prioritize_inat= hands
# to core as prefer= -- ranked first by the group cap, never removed by dedupe.
INAT_OCCURRENCES = {"institutionCode": ["iNaturalist"]}

DATETIME_COLS = [
    "eventDate",
    "modified",
    "dateIdentified",
    "lastInterpreted",
    "lastParsed",
    "lastCrawled",
]
NUMERIC_COLS = [
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "coordinatePrecision",
    "elevation",
    "elevationAccuracy",
    "depth",
    "depthAccuracy",
    "individualCount",
    "organismQuantity",
    "year",
    "month",
    "day",
]

# A practical occurrence_columns set for iNaturalist-via-GBIF ingests --
# narrows occurrence.txt from GBIF's 200+ columns down to what a
# citizen-science organism project typically keeps. NOT the default for
# occurrence_columns itself, which stays None ("read everything"): pass this
# explicitly, occurrence_columns=DEFAULT_INAT_OCCURRENCE_COLUMNS, to opt in.
# Includes occurrenceStatus, which the default drop=ABSENT_OCCURRENCES needs
# to see -- narrowing occurrence_columns without it raises (see
# ingest_occurrences' occurrence_columns docstring). Every name here is a
# standard GBIF interpreted field, present (if blank) in any processed
# occurrence.txt -- but pandas' usecols requires the column to actually be a
# header in the source, not just possibly empty, so an export missing one of
# these entirely (an unusual constituent dataset, an older archive format)
# raises rather than silently reading around it; drop the missing name from a
# copy of this tuple for that source instead.
DEFAULT_INAT_OCCURRENCE_COLUMNS = (
    "gbifID",
    "occurrenceID",
    "occurrenceStatus",
    "scientificName",
    "species",
    "genus",
    "family",
    "order",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "eventDate",
    "sex",
    "lifeStage",
    "identifiedBy",
    "dateIdentified",
    "identificationVerificationStatus",
    "elevation",
    "countryCode",
    "stateProvince",
    "county",
    "locality",
    "recordedBy",
    "recordedByID",
    "waterBody",
    "habitat",
    "license",
    "institutionCode",
)

# The renditions iNaturalist serves, and the two hosts it serves them from: the
# S3 bucket holding everything migrated to the Open Data program, and the older
# CDN domain some still-current URLs use. Anything else is somebody else's
# server -- Flickr, a museum's own image server, observation.org -- and
# rewriting its path would just break the URL.
INAT_PHOTO_SIZES = ("original", "large", "medium", "small", "square")
INAT_MEDIA_HOSTS = {
    "inaturalist-open-data.s3.amazonaws.com",
    "static.inaturalist.org",
}

# A ready-made cross-source duplicate fingerprint, for a project pulling from
# more than one GBIF-mediated source: the same real sighting, independently
# published by two aggregators (iNaturalist and Observation.org both feed
# GBIF, and a sighting logged on -- or mirrored to -- both shows up as two
# unrelated gbifIDs, so id-based dedup can't see it). Rounding lat/lon to 3
# decimal places is about 11m at the equator; eventDate is matched exactly
# since it's a source-populated date field, not a measurement. NOT the
# default for dedupe_key_cols -- pass this explicitly to opt in, the same way
# DEFAULT_INAT_OCCURRENCE_COLUMNS is opt-in rather than assumed.
DEFAULT_DEDUPE_KEY_COLS = ("decimalLatitude", "decimalLongitude", "eventDate")
DEFAULT_DEDUPE_PRECISION = {"decimalLatitude": 3, "decimalLongitude": 3}


def rewrite_inat_photo_size(url, size):
    """Rewrite an iNaturalist photo URL to another rendition, e.g. `original.jpg` to `medium.jpg`.

    A URL not served from an iNaturalist photo host, or a missing one, is returned unchanged.

    Args:
        url: A multimedia identifier value, or NaN.
        size: One of `INAT_PHOTO_SIZES`.
    """
    if size not in INAT_PHOTO_SIZES:
        raise ValueError(f"unknown iNaturalist photo size {size!r} -- use one of {INAT_PHOTO_SIZES}")
    if not isinstance(url, str) or not url:
        return url

    parsed = urlparse(url)
    if parsed.netloc not in INAT_MEDIA_HOSTS:
        return url

    path = PurePosixPath(parsed.path)
    if len(path.parts) < 2:
        return url

    return parsed._replace(path=str(path.with_name(f"{size}{path.suffix}"))).geturl()


def select_media(multimedia_df, rule="first", media_type=DEFAULT_MEDIA_TYPE):
    """Reduce a multimedia table to at most one row per occurrence.

    Rows are filtered to `media_type` before the rule picks one, so a sound recording
    can't be chosen over a later photo.

    Args:
        multimedia_df: The multimedia table.
        rule: `"first"` or `"last"` in the file's own order, or a
            `callable(group_df) -> row or None`.
        media_type: The multimedia `type` to require, e.g. `"StillImage"`; None keeps
            every row.

    Returns:
        One row per `gbifID` that had a matching row.
    """
    if media_type is not None:
        if TYPE_COL not in multimedia_df.columns:
            logger.warning("multimedia table has no '%s' column -- media_type filter skipped", TYPE_COL)
        else:
            before = len(multimedia_df)
            multimedia_df = multimedia_df[multimedia_df[TYPE_COL] == media_type]
            logger.info(
                "kept %d of %d multimedia row(s) with %s=%r", len(multimedia_df), before, TYPE_COL, media_type
            )

    if rule == "first":
        return multimedia_df.drop_duplicates(subset=[GBIF_ID_COL], keep="first")
    if rule == "last":
        return multimedia_df.drop_duplicates(subset=[GBIF_ID_COL], keep="last")
    if callable(rule):
        selected = [
            row
            for row in (rule(group) for _, group in multimedia_df.groupby(GBIF_ID_COL, sort=False))
            if row is not None
        ]
        if not selected:
            return multimedia_df.iloc[0:0]
        return pd.DataFrame(selected).reset_index(drop=True)

    raise ValueError(
        f"unknown media selection rule {rule!r} -- use 'first', "
        "'last', or a callable(group_df) -> one row or None"
    )


def merge_occurrence_media(occurrence_df, media_df):
    """Join one selected multimedia row onto each occurrence, and exclude occurrences with no image.

    Every multimedia column but `gbifID` gains a `media_` prefix. The number excluded is logged.

    Args:
        occurrence_df: The occurrence table.
        media_df: One multimedia row per occurrence, from `select_media`.
    """
    renamed = media_df.rename(
        columns={column: f"{MEDIA_PREFIX}{column}" for column in media_df.columns if column != GBIF_ID_COL}
    )
    merged = occurrence_df.merge(renamed, on=GBIF_ID_COL, how="left")

    has_image = merged[f"{MEDIA_PREFIX}{IDENTIFIER_COL}"].notna()
    excluded = int((~has_image).sum())
    if excluded:
        logger.info(
            "excluding %d of %d occurrence(s) with no usable image in this export", excluded, len(merged)
        )
    return merged[has_image].reset_index(drop=True)


def _raw_darwincore_bytes(occurrence_df, multimedia_df):
    """Zip the two tables back together as read, for tables handed in with no backing file.

    A re-serialization of what was parsed, covering only the columns read;
    `archive.raw_archive_bytes` is used whenever there is a real file.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("occurrence.csv", occurrence_df.to_csv(index=False))
        bundle.writestr("multimedia.csv", multimedia_df.to_csv(index=False))
    return buffer.getvalue()


def ingest_occurrences(
    project_path,
    archive_path=None,
    occurrence_df=None,
    multimedia_df=None,
    occurrence_columns=None,
    multimedia_columns=None,
    media_rule="first",
    media_type=DEFAULT_MEDIA_TYPE,
    inat_photo_size=None,
    transform=None,
    drop=ABSENT_OCCURRENCES,
    group_col=None,
    max_per_group=None,
    cap_rule="random",
    dedupe_key_cols=None,
    dedupe_precision=DEFAULT_DEDUPE_PRECISION,
    dedupe_rule="random",
    prioritize_inat=False,
    trust_source_file_unchanged=False,
    visualize=True,
):
    """Ingest a GBIF Darwin Core Archive into a project, as a full snapshot.

    The raw import archived is the two source tables as GBIF sent them, before one image
    is chosen per occurrence.

    Args:
        project_path: Project to ingest into; created if absent.
        archive_path: The archive: a `.zip`, or a directory it was extracted into.
        occurrence_df: A pre-read occurrence table, in place of `archive_path`.
        multimedia_df: A pre-read multimedia table, in place of `archive_path`.
        occurrence_columns: Columns of `occurrence.txt` to read; all if None. Must include
            every column `drop`, `group_col`, `transform` and `dedupe_key_cols` use.
            `DEFAULT_INAT_OCCURRENCE_COLUMNS` is a ready-made set.
        multimedia_columns: Columns of `multimedia.txt` to read; all if None.
        media_rule: How to pick one multimedia row per occurrence; see `select_media`.
        media_type: The multimedia `type` to require; None keeps sound and video too.
        inat_photo_size: `"original"`, `"medium"` or `"small"` rewrites iNaturalist photo
            URLs to that rendition; None leaves URLs as GBIF gives them.
        transform: A `callable(df) -> df` run after normalization.
        drop: `{column: values}` naming rows that are not organisms; `ABSENT_OCCURRENCES`
            by default. None keeps every row.
        group_col: As in `critterframe.ingest.ingest_occurrences`.
        max_per_group: As in `critterframe.ingest.ingest_occurrences`.
        cap_rule: As in `critterframe.ingest.ingest_occurrences`.
        dedupe_key_cols: Columns identifying one real sighting published more than once,
            e.g. `DEFAULT_DEDUPE_KEY_COLS`. Off if None. A missing column skips
            deduplication with a warning.
        dedupe_precision: `{column: ndigits}` to round a numeric key column by.
        dedupe_rule: Which row of a duplicate group survives.
        prioritize_inat: Fill each group's cap from iNaturalist rows first, and let
            deduplication remove only other rows. Needs `institutionCode` among
            `occurrence_columns`.
        trust_source_file_unchanged: As in `critterframe.ingest.ingest_occurrences`.
            Applies only when `archive_path` is a `.zip`.
        visualize: Write pipeline figures of the rows kept at each stage.

    Returns:
        The resulting occurrence table.
    """
    logger.info("starting GBIF ingest into %s (archive_path=%s)", project_path, archive_path)

    if prioritize_inat and not (group_col or dedupe_key_cols):
        logger.warning(
            "prioritize_inat has nothing to act on without group_col/max_per_group or dedupe_key_cols"
        )

    if archive_path is not None:
        if occurrence_df is not None or multimedia_df is not None:
            raise ValueError("pass either archive_path or occurrence_df/multimedia_df, not both")
        source = archive_path
        name_prefix = f"occurrences_gbif_{Path(archive_path).stem}"

        def read_archive(path):
            return _build(
                *archive.read_darwincore_archive(
                    path, occurrence_usecols=occurrence_columns, multimedia_usecols=multimedia_columns
                )
            )

        read, raw = read_archive, archive.raw_archive_bytes
    elif occurrence_df is not None and multimedia_df is not None:
        # A synthetic name: nothing backs it on disk, so it is never trusted
        # on a fingerprint, and the raw import is the two tables zipped.
        source = "occurrences_gbif.csv"
        name_prefix = "occurrences_gbif"

        def read_tables(_):
            return _build(occurrence_df, multimedia_df)

        def zip_tables(_):
            return _raw_darwincore_bytes(occurrence_df, multimedia_df), ".zip"

        read, raw = read_tables, zip_tables
    else:
        raise ValueError("pass archive_path, or both occurrence_df and multimedia_df")

    def _build(occurrences, multimedia):
        """One media row per occurrence, merged on, with the stage counts."""
        logger.info(
            "picking one multimedia row per occurrence from %d row(s) (media_rule=%r, media_type=%r)",
            len(multimedia),
            media_rule,
            media_type,
        )
        with timed("picked media rows", logger.info) as done:
            media = select_media(multimedia, rule=media_rule, media_type=media_type)
            done["rows"] = len(media)

        if inat_photo_size is not None:
            logger.info("rewriting %d photo URL(s) to inat_photo_size=%r", len(media), inat_photo_size)
            with timed("rewrote photo URLs", logger.info):
                media = media.copy()
                media[IDENTIFIER_COL] = media[IDENTIFIER_COL].map(
                    lambda url: rewrite_inat_photo_size(url, inat_photo_size)
                )

        logger.info("merging %d occurrence(s) with their picked media", len(occurrences))
        with timed("merged", logger.info) as done:
            merged = merge_occurrence_media(occurrences, media)
            done["rows"] = len(merged)

        return merged, {
            "occurrence rows": len(occurrences),
            "media rows picked": len(media),
            "with an image": len(merged),
        }

    # Core runs read only once the import is known to be new, so an unchanged
    # archive is never parsed twice. transform is passed straight through, not
    # wrapped: a closure would record one fixed name whatever it wrapped, and
    # two different transforms would then share an import hash.
    return core_ingest.ingest_occurrences(
        project_path,
        source,
        id_col=GBIF_ID_COL,
        image_url_col=f"{MEDIA_PREFIX}{IDENTIFIER_COL}",
        datetime_cols=DATETIME_COLS,
        numeric_cols=NUMERIC_COLS,
        transform=transform,
        drop=drop,
        group_col=group_col,
        max_per_group=max_per_group,
        cap_rule=cap_rule,
        dedupe_key_cols=dedupe_key_cols,
        dedupe_precision=dedupe_precision,
        dedupe_rule=dedupe_rule,
        prefer=INAT_OCCURRENCES if prioritize_inat else None,
        name_prefix=name_prefix,
        manifest_extra=_manifest_extra(
            archive_path,
            occurrence_columns,
            multimedia_columns,
            media_rule,
            media_type,
            inat_photo_size,
            dedupe_key_cols,
            dedupe_precision,
            dedupe_rule,
        ),
        read=read,
        raw=raw,
        trust_source_file_unchanged=trust_source_file_unchanged,
        visualize=visualize,
    )


def _manifest_extra(
    archive_path,
    occurrence_columns,
    multimedia_columns,
    media_rule,
    media_type,
    inat_photo_size,
    dedupe_key_cols,
    dedupe_precision,
    dedupe_rule,
):
    """Return the GBIF-specific decisions recorded in the import manifest and its hash."""
    return {
        "source": str(archive_path) if archive_path is not None else "occurrence_df/multimedia_df",
        "occurrence_columns": occurrence_columns,
        "multimedia_columns": multimedia_columns,
        "media_rule": media_rule
        if isinstance(media_rule, str)
        else getattr(media_rule, "__qualname__", "<callable>"),
        "media_type": media_type,
        "inat_photo_size": inat_photo_size,
        # Kept here as well as in core's own recipe (which records them only
        # when deduplication is on): dropping them would move the import hash
        # of every archive ingested before deduplication became core's, and
        # re-reading a multi-gigabyte export to reach the same table is a poor
        # trade for tidiness.
        "dedupe_key_cols": list(dedupe_key_cols) if dedupe_key_cols else None,
        "dedupe_precision": dedupe_precision,
        "dedupe_rule": recorded_callable(dedupe_rule),
    }
