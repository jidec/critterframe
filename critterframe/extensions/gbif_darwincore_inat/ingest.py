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

Scale is unrecoverable here: there
is no reference object in a citizen-science photo, so every trait measured
from a GBIF-ingested project is in pixels.

The raw import core_ingest.ingest_occurrences archives is the two source
tables as GBIF sent them, zipped together, BEFORE select_media picks one
photo per occurrence -- not the merged, one-row-per-occurrence CSV built
here. Which photo of several represents an occurrence is a real choice
(media_rule), not a structural given, so it belongs in the import manifest as
a decision rather than being silently baked into what's archived as "raw".

The parse is handed to core as read=, which core runs only once the import
is known to be new, so an unchanged archive is never parsed twice.
"""

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
    """
    Rewrite an iNaturalist photo URL to request a different rendition, e.g.
    ".../photos/404810759/original.jpg" -> ".../photos/404810759/medium.jpg".

    Any URL not served from an iNaturalist photo host comes back unchanged, and
    so does a missing one (None or NaN).

    - `url` -- a multimedia identifier value (or NaN).
    - `size` -- one of `INAT_PHOTO_SIZES`.
    """
    if size not in INAT_PHOTO_SIZES:
        raise ValueError(f"unknown iNaturalist photo size {size!r} -- use one "
                         f"of {INAT_PHOTO_SIZES}")
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
                       dedupe_rule="random", prioritize_inat=False,
                       trust_source_file_unchanged=False, visualize=True):
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
      `critterframe.ingest.ingest_occurrences`, which applies it after
      `drop=`, and `selectionhelpers.dedupe_by` for what these mean, and
      `DEFAULT_DEDUPE_KEY_COLS`/`DEFAULT_DEDUPE_PRECISION` for a ready-made
      (`decimalLatitude`, `decimalLongitude`, `eventDate`) fingerprint
      (~11m on the coordinates, exact on the date) to pass for
      `dedupe_key_cols`. Only image-bearing rows ever reach it, since
      `merge_occurrence_media` has already run. This is a probabilistic
      match, not a source-declared fact like `drop=` -- loosen
      `dedupe_precision` only as far as the real risk of a false match
      warrants.
    - `prioritize_inat` -- True: fill each `max_per_group` cap from
      iNaturalist rows (`INAT_OCCURRENCES`, i.e.
      `institutionCode="iNaturalist"`) first, drawing from other sources only
      when a group has too few, and let deduplication remove only non-iNat
      rows. Ranks above occurrences the project already holds, so a later
      pull with more iNat rows displaces non-iNat specimens kept earlier.
      `institutionCode` must survive `occurrence_columns`. Does nothing
      without `group_col` or `dedupe_key_cols`. False (default) treats every
      source alike.
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
    - `visualize` -- True (default): pipeline figures of the rows kept at
      each stage, from the archive's own tables through media selection,
      deduplication, and core's drop=/cap. False writes nothing.

    Returns the resulting occurrence table.
    """
    logger.info("starting GBIF ingest into %s (archive_path=%s)",
               project_path, archive_path)

    if prioritize_inat and not (group_col or dedupe_key_cols):
        logger.warning("prioritize_inat has nothing to act on without "
                       "group_col/max_per_group or dedupe_key_cols")

    if archive_path is not None:
        if occurrence_df is not None or multimedia_df is not None:
            raise ValueError("pass either archive_path or occurrence_df/"
                             "multimedia_df, not both")
        source = archive_path
        name_prefix = f"occurrences_gbif_{Path(archive_path).stem}"

        def read_archive(path):
            return _build(*archive.read_darwincore_archive(
                path, occurrence_usecols=occurrence_columns,
                multimedia_usecols=multimedia_columns))

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
        logger.info("picking one multimedia row per occurrence from %d row(s) "
                   "(media_rule=%r, media_type=%r)", len(multimedia),
                   media_rule, media_type)
        with timed("picked media rows", logger.info) as done:
            media = select_media(multimedia, rule=media_rule, media_type=media_type)
            done["rows"] = len(media)

        if inat_photo_size is not None:
            logger.info("rewriting %d photo URL(s) to inat_photo_size=%r",
                       len(media), inat_photo_size)
            with timed("rewrote photo URLs", logger.info):
                media = media.copy()
                media[IDENTIFIER_COL] = media[IDENTIFIER_COL].map(
                    lambda url: rewrite_inat_photo_size(url, inat_photo_size))

        logger.info("merging %d occurrence(s) with their picked media", len(occurrences))
        with timed("merged", logger.info) as done:
            merged = merge_occurrence_media(occurrences, media)
            done["rows"] = len(merged)

        return merged, {"occurrence rows": len(occurrences),
                        "media rows picked": len(media),
                        "with an image": len(merged)}

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
        manifest_extra=_manifest_extra(archive_path, occurrence_columns,
                                       multimedia_columns, media_rule,
                                       media_type, inat_photo_size,
                                       dedupe_key_cols, dedupe_precision,
                                       dedupe_rule),
        read=read,
        raw=raw,
        trust_source_file_unchanged=trust_source_file_unchanged,
        visualize=visualize,
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
        # Kept here as well as in core's own recipe (which records them only
        # when deduplication is on): dropping them would move the import hash
        # of every archive ingested before deduplication became core's, and
        # re-reading a multi-gigabyte export to reach the same table is a poor
        # trade for tidiness.
        "dedupe_key_cols": list(dedupe_key_cols) if dedupe_key_cols else None,
        "dedupe_precision": dedupe_precision,
        "dedupe_rule": recorded_callable(dedupe_rule),
    }
