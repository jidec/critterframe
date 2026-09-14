"""
Normalize and ingest a GBIF Darwin Core Archive into a CritterFrame project.

GBIF splits one occurrence across two tables: occurrence.txt (everything about
the record) and multimedia.txt (zero or more media items per record, joined
back by gbifID). CritterFrame needs the reverse -- one row, with one image
URL -- so this picks a single multimedia row per occurrence (select_media) and
merges it onto the occurrence row (merge_occurrence_media) before handing the
result to the core ingest.

That two-table shape is universal to GBIF, whatever the original data
provider was. The one thing here that ISN'T universal is
rewrite_inat_photo_size: many GBIF occurrences carry iNaturalist photo URLs
(observations imported from inaturalist.org, or occasionally re-published
through another aggregator), and iNaturalist serves several renditions of the
same photo from the same URL shape. Every other identifier -- a museum's own
image server, Flickr, observation.org -- passes through untouched.

Scale is unrecoverable here for the same reason it is for inat_insects: there
is no reference object in a citizen-science photo, so every trait measured
from a GBIF-ingested project is in pixels.

The raw import core_ingest.ingest_occurrences archives is the two source
tables as GBIF sent them, zipped together, BEFORE select_media picks one
photo per occurrence -- not the merged, one-row-per-occurrence CSV built
here. Which photo of several represents an occurrence is a real choice
(media_rule), not a structural given, so it belongs in the import manifest as
a decision rather than being silently baked into what's archived as "raw".

already_ingested is checked against archive_path's raw bytes BEFORE
read_darwincore_archive parses it, not after: for a multi-gigabyte export
that parse is itself the expensive step, so skipping it is what makes a
repeat ingest of an unchanged pull actually cheap, rather than only skipping
core_ingest.ingest_occurrences' own write once the parse has already run.
"""

import io
import logging
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd

from ... import ingest as core_ingest
from ...records import occurrences as occurrence_records
from ... import selectionhelpers
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

DATETIME_COLS = [
    "eventDate", "modified", "dateIdentified",
    "lastInterpreted", "lastParsed", "lastCrawled",
]
NUMERIC_COLS = [
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "coordinatePrecision", "elevation", "elevationAccuracy", "depth",
    "depthAccuracy", "individualCount", "organismQuantity", "year", "month", "day",
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
    "gbifID", "occurrenceID", "occurrenceStatus",
    "scientificName", "species", "genus", "family", "order",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "eventDate", "sex", "lifeStage",
    "identifiedBy", "dateIdentified", "identificationVerificationStatus",
    "elevation", "countryCode", "stateProvince", "county", "locality",
    "recordedBy", "recordedByID", "waterBody", "habitat", "license","institutionCode"
)

# Hosts iNaturalist actually serves photo renditions from -- the S3 bucket
# holding everything migrated to the Open Data program, and the older CDN
# domain some still-current URLs use. Anything else is somebody else's server
# and rewriting its path would just break the URL.
INAT_MEDIA_HOSTS = {
    "inaturalist-open-data.s3.amazonaws.com",
    "static.inaturalist.org",
}
INAT_PHOTO_SIZES = ("original", "medium", "small")

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
    """
    Rewrite an iNaturalist photo URL to request a different rendition, e.g.
    ".../photos/404810759/original.jpg" -> ".../photos/404810759/medium.jpg".

    Any URL not served from an iNaturalist photo host is returned unchanged --
    this dataset mixes providers freely, and rewriting a path GBIF got from
    Flickr or a museum server would just break it.

    - `url` -- a multimedia identifier value (or NaN).
    - `size` -- `"original"`, `"medium"`, or `"small"`.
    """
    if size not in INAT_PHOTO_SIZES:
        raise ValueError(f"unknown iNaturalist photo size {size!r} -- use one "
                         f"of {INAT_PHOTO_SIZES}")
    if pd.isna(url) or not isinstance(url, str):
        return url

    parsed = urlparse(url)
    if parsed.netloc not in INAT_MEDIA_HOSTS:
        return url

    old_path = PurePosixPath(parsed.path)
    if len(old_path.parts) < 2:
        return url

    new_path = str(old_path.with_name(f"{size}{old_path.suffix}"))
    return parsed._replace(path=new_path).geturl()


def select_media(multimedia_df, rule="first", media_type=DEFAULT_MEDIA_TYPE):
    """
    Reduce a multimedia table to at most one row per occurrence.

    Filtering to media_type happens BEFORE the one-per-occurrence rule, not
    after -- multimedia.txt interleaves photos with sound recordings and video
    (iNaturalist publishes both through GBIF), and taking "first" before
    filtering could hand back a sound recording for an occurrence whose actual
    first photo sorts later in the file.

    - `multimedia_df` -- the multimedia table, as
      `archive.read_darwincore_table` returns it.
    - `rule` -- `"first"` (default) or `"last"` takes the first/last
      matching row per occurrence, in the file's own order; a
      `callable(group_df) -> one row` (a Series) or None picks explicitly,
      e.g. by license or resolution.
    - `media_type` -- the multimedia `type` value to require, e.g.
      `"StillImage"`. None keeps every row regardless of type.

    Returns one row per gbifID that had a matching row. An occurrence with none
    is simply absent from the result -- merge_occurrence_media reports how
    many that was, rather than this raising.
    """
    if media_type is not None:
        if TYPE_COL not in multimedia_df.columns:
            logger.warning("multimedia table has no '%s' column -- media_type "
                           "filter skipped", TYPE_COL)
        else:
            before = len(multimedia_df)
            multimedia_df = multimedia_df[multimedia_df[TYPE_COL] == media_type]
            logger.info("kept %d of %d multimedia row(s) with %s=%r",
                       len(multimedia_df), before, TYPE_COL, media_type)

    if rule == "first":
        return multimedia_df.drop_duplicates(subset=[GBIF_ID_COL], keep="first")
    if rule == "last":
        return multimedia_df.drop_duplicates(subset=[GBIF_ID_COL], keep="last")
    if callable(rule):
        selected = [row for row in (
            rule(group) for _, group in multimedia_df.groupby(GBIF_ID_COL, sort=False))
            if row is not None]
        if not selected:
            return multimedia_df.iloc[0:0]
        return pd.DataFrame(selected).reset_index(drop=True)

    raise ValueError(f"unknown media selection rule {rule!r} -- use 'first', "
                     "'last', or a callable(group_df) -> one row or None")


def merge_occurrence_media(occurrence_df, media_df):
    """
    Left-join one selected multimedia row onto each occurrence, then exclude
    any occurrence left without a usable image.

    Not ABSENT_OCCURRENCES' job: GBIF hasn't said the organism is absent, so
    this isn't ingest_occurrences' drop= contract. It's this project deciding
    that an occurrence with no image can't be an occurrence-per-image record
    at all -- a usability decision about this ingest, not a source-declared
    category -- so it's made here instead. Every excluded row is still fully
    recoverable from the archived raw import (occurrence.txt + multimedia.txt
    exactly as GBIF sent them); only the aggregate count is logged, no
    per-row manifest, the same convention selectionhelpers.cap_per_group uses
    for rows it removes.

    Every multimedia column except gbifID is renamed with a `media_` prefix
    first, since occurrence.txt and multimedia.txt both define columns named
    type, license, and rightsHolder for different things.
    """
    renamed = media_df.rename(columns={
        column: f"{MEDIA_PREFIX}{column}" for column in media_df.columns
        if column != GBIF_ID_COL
    })
    merged = occurrence_df.merge(renamed, on=GBIF_ID_COL, how="left")

    has_image = merged[f"{MEDIA_PREFIX}{IDENTIFIER_COL}"].notna()
    excluded = int((~has_image).sum())
    if excluded:
        logger.info("excluding %d of %d occurrence(s) with no usable image "
                    "in this export", excluded, len(merged))
    return merged[has_image].reset_index(drop=True)


def _raw_darwincore_bytes(occurrence_df, multimedia_df):
    """
    Zip occurrence.txt and multimedia.txt back together, exactly as read,
    before select_media makes any choice about which photo represents an
    occurrence.

    Fallback for when occurrence_df/multimedia_df were handed in directly,
    with no backing file for archive.raw_archive_bytes to copy -- these are
    a re-serialization of what was parsed, not the original bytes, and only
    ever cover the columns usecols narrowed the read to. Whenever there IS a
    real file (archive_path), raw_archive_bytes is used instead: faster, and
    genuinely byte-exact regardless of what a particular ingest read.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("occurrence.csv", occurrence_df.to_csv(index=False))
        bundle.writestr("multimedia.csv", multimedia_df.to_csv(index=False))
    return buffer.getvalue()


def ingest_occurrences(project_path, archive_path=None, occurrence_df=None,
                       multimedia_df=None, occurrence_columns=None,
                       multimedia_columns=None, media_rule="first",
                       media_type=DEFAULT_MEDIA_TYPE, inat_photo_size=None,
                       transform=None, drop=ABSENT_OCCURRENCES, group_col=None,
                       max_per_group=None, cap_rule="random",
                       dedupe_key_cols=None,
                       dedupe_precision=DEFAULT_DEDUPE_PRECISION,
                       dedupe_rule="random",
                       trust_source_file_unchanged=False):
    """
    Ingest a GBIF Darwin Core Archive into a project, as a full snapshot.

    Works for any GBIF-published occurrence data; inat_photo_size is the one
    iNaturalist-specific option and does nothing for anything else.

    - `project_path` -- project to ingest into; created lazily.
    - `archive_path` -- the archive as GBIF hands it out: a .zip file, or a
      directory it's already been extracted into. Omit and pass
      `occurrence_df`/`multimedia_df` instead when the tables are already
      loaded.
    - `occurrence_df`, `multimedia_df` -- pre-read tables
      (`archive.read_darwincore_table`), used instead of `archive_path`.
    - `occurrence_columns`, `multimedia_columns` -- narrow either table to
      these columns while reading `archive_path`; ignored when
      `occurrence_df`/`multimedia_df` are passed in already read. A
      processed `occurrence.txt` is 200+ columns wide, every one
      individually allocated under `dtype=str` -- for a multi-gigabyte
      export this is the difference between fitting in memory and not.
      Must include every column `drop=`, `group_col`, `transform`, and
      `dedupe_key_cols` touch -- the default `drop=ABSENT_OCCURRENCES`
      needs `"occurrenceStatus"`, and `dedupe_key_cols` (off by default)
      needs whatever columns it names if turned on; narrowing them out is
      not an error either way -- a missing `dedupe_key_cols` column skips
      deduplication with a warning rather than failing the ingest -- and
      every column downstream code reads off the final table; `gbifID` is
      always kept regardless, since the merge needs it. None (default)
      reads every column, as before. See `DEFAULT_INAT_OCCURRENCE_COLUMNS`
      for a ready-made set covering a typical iNaturalist-via-GBIF project.
    - `media_rule` -- how to pick one multimedia row per occurrence when
      there's more than one; see `select_media`.
    - `media_type` -- multimedia `type` to require, e.g. `"StillImage"`.
      None keeps every row, including sound and video.
    - `inat_photo_size` -- `"original"`, `"medium"`, or `"small"` rewrites
      iNaturalist photo URLs to that rendition before anything else runs.
      None (default) leaves every URL exactly as GBIF gives it.
    - `transform` -- optional `callable(df) -> df` run after normalization.
    - `drop` -- `{column: values}` naming rows the source says aren't
      organisms; defaults to `ABSENT_OCCURRENCES`, since GBIF's
      `occurrenceStatus="ABSENT"` is exactly that. Pass None to keep every
      row.
    - `group_col`, `max_per_group`, `cap_rule` -- cap ingest at
      `max_per_group` rows per distinct value of `group_col`, e.g.
      `group_col="species"`, `max_per_group=200` against a download
      dominated by a few common species. See
      `critterframe.ingest.ingest_occurrences`. Applied after `drop=`, so
      an ABSENT record never counts toward its group's cap.
    - `dedupe_key_cols`, `dedupe_precision`, `dedupe_rule` -- OFF by
      default (`dedupe_key_cols=None`); opt in to fold together rows that
      are the same real sighting published more than once -- iNaturalist
      and Observation.org both feed GBIF, and one sighting logged on (or
      mirrored to) both gets two unrelated gbifIDs, so no id-based check
      can catch it. Only worth turning on for a project actually combining
      more than one source; skip it and filter to one source with
      `transform=` instead when that's all a project needs. See
      `selectionhelpers.dedupe_by` for what these mean, and
      `DEFAULT_DEDUPE_KEY_COLS`/`DEFAULT_DEDUPE_PRECISION` for a ready-made
      (`decimalLatitude`, `decimalLongitude`, `eventDate`) fingerprint
      (~11m on the coordinates, exact on the date) to pass for
      `dedupe_key_cols`. Applied to the image-bearing rows AFTER
      `merge_occurrence_media` -- narrower and cheaper than deduping the
      full occurrence table, and ahead of `group_col`/`max_per_group` so a
      per-species cap counts real specimens rather than inflated
      duplicates. This is a probabilistic match, not a source-declared
      fact like `drop=` -- loosen `dedupe_precision` only as far as the
      real risk of a false match warrants.
    - `trust_source_file_unchanged` -- skip copying and hashing
      `archive_path`'s content when its path, size, and modification time
      exactly match a previous import already on record for this project,
      and reuse that import's `raw_hash` instead. For a multi-gigabyte
      GBIF export, this is the difference between a repeat, no-op ingest
      finishing instantly and it re-copying and re-hashing the whole
      archive just to discover nothing changed. Only applies when
      `archive_path` is a .zip -- an already-extracted directory always
      falls back to a real read, since a directory's size isn't a
      meaningful content proxy. This is NOT a content guarantee:
      path+size+mtime matching does not prove the bytes are unchanged. A
      file regenerated with coincidentally the same size, a
      touched-but-otherwise-identical file, clock skew, or a different
      file copied over the same path with the same size, would all be
      (wrongly) trusted. Off by default for exactly that reason. Every
      OTHER decision (`occurrence_columns`, `drop=`, `group_col=`, etc.)
      is still checked exactly as always, so changing one of those against
      an unchanged archive still triggers a real re-ingest rather than
      being skipped.

    Returns the resulting occurrence table.
    """
    logger.info("starting GBIF ingest into %s (archive_path=%s)",
               project_path, archive_path)

    if archive_path is not None:
        if occurrence_df is not None or multimedia_df is not None:
            raise ValueError("pass either archive_path or occurrence_df/"
                             "multimedia_df, not both")
        name_prefix = f"occurrences_gbif_{Path(archive_path).stem}"
        manifest_extra = _manifest_extra(archive_path, occurrence_columns,
                                         multimedia_columns, media_rule,
                                         media_type, inat_photo_size,
                                         dedupe_key_cols, dedupe_precision,
                                         dedupe_rule)

        # A .zip is copied/hashed as bytes, an extracted directory is
        # re-zipped from disk (see archive.raw_archive_bytes) -- proportional
        # to archive size either way, minutes for a multi-gigabyte export.
        # Deliberately done BEFORE the expensive read_darwincore_archive parse
        # below, so the already_ingested check that follows can skip that
        # parse entirely on a repeat run, rather than only skipping the write
        # that would otherwise happen after paying for it anyway.
        # trust_source_file_unchanged=True skips this read+hash itself, not
        # just the parse, when a cheap path/size/mtime fingerprint already
        # answers the question.
        if trust_source_file_unchanged and zipfile.is_zipfile(archive_path):
            if core_ingest.fingerprint_unchanged(
                    project_path, archive_path, id_col=GBIF_ID_COL,
                    image_url_col=f"{MEDIA_PREFIX}{IDENTIFIER_COL}",
                    datetime_cols=DATETIME_COLS, numeric_cols=NUMERIC_COLS,
                    transform=transform, drop=drop, group_col=group_col,
                    max_per_group=max_per_group, cap_rule=cap_rule,
                    manifest_extra=manifest_extra):
                logger.info("trusting %s is unchanged (same path, size, and "
                            "mtime as a previous import) -- skipping the "
                            "copy+hash of the archive entirely", archive_path)
                return occurrence_records.load_occurrences(project_path)

        logger.info("copying %s as the raw import, unparsed", archive_path)
        start = time.monotonic()
        raw_bytes, raw_extension = archive.raw_archive_bytes(archive_path)
        logger.info("copied raw import in %.1fs (%.1f MB)",
                   time.monotonic() - start, len(raw_bytes) / 1e6)

        if core_ingest.already_ingested(
                project_path, raw_bytes, id_col=GBIF_ID_COL,
                image_url_col=f"{MEDIA_PREFIX}{IDENTIFIER_COL}",
                datetime_cols=DATETIME_COLS, numeric_cols=NUMERIC_COLS,
                transform=transform, drop=drop, group_col=group_col,
                max_per_group=max_per_group, cap_rule=cap_rule,
                manifest_extra=manifest_extra):
            logger.info("this raw import has already been ingested with these "
                        "same decisions -- skipping the parse of %s entirely",
                        archive_path)
            return occurrence_records.load_occurrences(project_path)

        occurrence_df, multimedia_df = archive.read_darwincore_archive(
            archive_path, occurrence_usecols=occurrence_columns,
            multimedia_usecols=multimedia_columns)
    elif occurrence_df is not None and multimedia_df is not None:
        name_prefix = "occurrences_gbif"

        logger.info("zipping the raw occurrence/multimedia tables (%d + %d "
                   "rows) for archiving -- no backing file to copy",
                   len(occurrence_df), len(multimedia_df))
        start = time.monotonic()
        raw_bytes = _raw_darwincore_bytes(occurrence_df, multimedia_df)
        raw_extension = ".zip"
        logger.info("zipped raw import in %.1fs (%.1f MB)",
                   time.monotonic() - start, len(raw_bytes) / 1e6)

        # No already_ingested check here: occurrence_df/multimedia_df are
        # already-parsed tables the caller built, so there is no expensive
        # parse left to skip -- ingest_occurrences' own idempotency check
        # covers this case exactly as well.
        manifest_extra = _manifest_extra(None, occurrence_columns,
                                         multimedia_columns, media_rule,
                                         media_type, inat_photo_size,
                                         dedupe_key_cols, dedupe_precision,
                                         dedupe_rule)
    else:
        raise ValueError("pass archive_path, or both occurrence_df and multimedia_df")

    logger.info("picking one multimedia row per occurrence from %d row(s) "
               "(media_rule=%r, media_type=%r)", len(multimedia_df),
               media_rule, media_type)
    start = time.monotonic()
    media = select_media(multimedia_df, rule=media_rule, media_type=media_type)
    logger.info("picked %d row(s) in %.1fs", len(media), time.monotonic() - start)

    if inat_photo_size is not None:
        logger.info("rewriting %d photo URL(s) to inat_photo_size=%r",
                   len(media), inat_photo_size)
        start = time.monotonic()
        media = media.copy()
        media[IDENTIFIER_COL] = media[IDENTIFIER_COL].map(
            lambda url: rewrite_inat_photo_size(url, inat_photo_size))
        logger.info("rewrote photo URLs in %.1fs", time.monotonic() - start)

    logger.info("merging %d occurrence(s) with their picked media", len(occurrence_df))
    start = time.monotonic()
    merged = merge_occurrence_media(occurrence_df, media)
    logger.info("merged in %.1fs", time.monotonic() - start)

    # After merge_occurrence_media, not before: only image-bearing rows can
    # ever become occurrences, so there's no reason to fingerprint the ones
    # that are about to be excluded anyway. Before group_col/max_per_group
    # (in core, below), so a per-species cap counts real specimens rather
    # than a sighting inflated by however many aggregators published it.
    #
    # A column entirely ABSENT from merged (as opposed to merely blank on some
    # rows, which dedupe_by already exempts row-by-row) means this export, or
    # an occurrence_columns= that narrowed it out, never carried it -- not a
    # caller mistake worth failing a multi-gigabyte ingest over, unlike an
    # unknown column passed to dedupe_by directly.
    missing_dedupe_cols = [c for c in dedupe_key_cols if c not in merged.columns] \
        if dedupe_key_cols else []
    if missing_dedupe_cols:
        logger.warning("skipping deduplication -- %s not in this export "
                       "(narrowed out by occurrence_columns=, or absent from "
                       "the source)", missing_dedupe_cols)
    elif dedupe_key_cols:
        logger.info("deduplicating %d occurrence(s) on %s", len(merged),
                   ", ".join(dedupe_key_cols))
        start = time.monotonic()
        existing_ids = occurrence_records.load_occurrences(
            project_path, columns=[], missing_ok=True)[occurrence_records.ID_COL]
        merged = selectionhelpers.dedupe_by(
            merged, dedupe_key_cols, precision=dedupe_precision,
            rule=dedupe_rule, id_col=GBIF_ID_COL, keep_ids=set(existing_ids))
        logger.info("deduplicated in %.1fs (%d remain)", time.monotonic() - start,
                   len(merged))

    # merged is handed to core directly (df=), rather than written to a CSV
    # for core to re-read: for a multi-gigabyte, dtype=str source, that
    # round trip would cost a full extra write and parse, AND re-run pandas'
    # type inference over every column not explicitly coerced below --
    # silently undoing the very "left as strings" guarantee
    # read_darwincore_table exists for. import_csv_path is passed only as a
    # name to log and to derive the archived raw import's extension from.
    #
    # transform is passed straight through, not wrapped -- core already
    # treats None as "no transform", and a wrapper closure would be a fresh
    # function object every call, recording the SAME fixed name in the
    # manifest regardless of what transform actually was and making
    # already_ingested (and core's own idempotency check) unable to tell two
    # different transforms apart.
    return core_ingest.ingest_occurrences(
        project_path,
        f"{name_prefix}.csv",
        id_col=GBIF_ID_COL,
        image_url_col=f"{MEDIA_PREFIX}{IDENTIFIER_COL}",
        datetime_cols=DATETIME_COLS,
        numeric_cols=NUMERIC_COLS,
        transform=transform,
        drop=drop,
        group_col=group_col,
        max_per_group=max_per_group,
        cap_rule=cap_rule,
        name_prefix=name_prefix,
        raw_bytes=raw_bytes,
        raw_extension=raw_extension,
        manifest_extra=manifest_extra,
        df=merged,
        trust_source_file_unchanged=trust_source_file_unchanged,
        fingerprint_source_path=archive_path,
    )


def _manifest_extra(archive_path, occurrence_columns, multimedia_columns,
                    media_rule, media_type, inat_photo_size,
                    dedupe_key_cols, dedupe_precision, dedupe_rule):
    """The gbif-specific decisions folded into the import manifest and its
    hash, alongside core's own -- see ingest_occurrences' manifest_extra."""
    return {
        "source": str(archive_path) if archive_path is not None
                  else "occurrence_df/multimedia_df",
        "occurrence_columns": occurrence_columns,
        "multimedia_columns": multimedia_columns,
        "media_rule": media_rule if isinstance(media_rule, str)
                     else getattr(media_rule, "__qualname__", "<callable>"),
        "media_type": media_type,
        "inat_photo_size": inat_photo_size,
        "dedupe_key_cols": list(dedupe_key_cols) if dedupe_key_cols else None,
        "dedupe_precision": dedupe_precision,
        "dedupe_rule": dedupe_rule if isinstance(dedupe_rule, str)
                      else getattr(dedupe_rule, "__qualname__", "<callable>"),
    }
