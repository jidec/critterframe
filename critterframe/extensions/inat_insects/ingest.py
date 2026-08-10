"""
Normalize and ingest iNaturalist observations into a CritterFrame project.

The mapping is one observation to one occurrence, taking its FIRST photo as the
analysis image. That's the rule the data model requires -- one focal organism
per image -- and taking every photo of an observation would break it by
creating several occurrences of the same individual, which would then be
counted several times in any population-level statistic and, worse, would leak
the same animal into both sides of a training split.

Two caveats worth stating outright, because iNaturalist data does not behave
like specimen photography:

  A photo may contain more than one organism, or none clearly. Nothing here can
  tell. That's what the QC metrics and human screening are for -- run
  metrics.quality and metrics.annotation over an iNaturalist project before
  trusting its traits, more so than for a controlled imaging setup.

  Scale is unrecoverable. There is no reference object, so every trait from an
  iNaturalist project is in pixels and comparable only as a ratio or within a
  fixed photographic setup that doesn't exist here. Shape and colour traits
  survive this; absolute size does not.

The observation payload is archived as CSV before ingest like any other import,
so a project can always be rebuilt from what the API actually returned rather
than from what it would return today.
"""

import logging

import pandas as pd

from ...ingest import ingest_occurrences as core_ingest_occurrences
from ...project import paths
from . import api

logger = logging.getLogger(__name__)

DATETIME_COLS = ["observed_on", "created_at"]
NUMERIC_COLS = ["latitude", "longitude", "positional_accuracy",
                "identifications_count"]


def observation_row(observation, photo_size=api.PHOTO_SIZE):
    """
    Flatten one observation into the columns a project keeps.

    Deliberately narrow. The API returns a deep nested payload, and carrying all
    of it into a parquet column-per-field would produce an unreadable table --
    what's kept is identity, image, taxonomy, place, time, and license, which
    is what grouping, filtering, and attribution actually need.
    """
    taxon = observation.get("taxon") or {}
    coordinates = observation.get("geojson") or {}
    longitude, latitude = (coordinates.get("coordinates") or [None, None])

    return {
        "occurrence_id": str(observation["id"]),
        "image_url": api.photo_url(observation, size=photo_size),
        "observation_url": observation.get("uri"),
        "taxon": api.taxon_name(observation, rank="species"),
        "taxon_id": taxon.get("id"),
        "taxon_rank": taxon.get("rank"),
        "genus": api.taxon_name(observation, rank="genus"),
        "family": api.taxon_name(observation, rank="family"),
        "observed_on": observation.get("observed_on"),
        "created_at": observation.get("created_at"),
        "place_guess": observation.get("place_guess"),
        "latitude": latitude,
        "longitude": longitude,
        "positional_accuracy": observation.get("positional_accuracy"),
        "quality_grade": observation.get("quality_grade"),
        "identifications_count": observation.get("identifications_count"),
        "license_code": observation.get("license_code"),
        "observer": (observation.get("user") or {}).get("login"),
    }


def fetch_observations(project_path, taxon_name=None, place_id=None, limit=None,
                       photo_size=api.PHOTO_SIZE, session=None, **search_kwargs):
    """
    Search iNaturalist and write the results to a CSV in the project's imports
    directory, returning its path.

    Separate from ingesting so a long pull is done once and can be re-ingested
    without hitting the API again -- worth having when a query takes an hour of
    rate-limited requests and the ingest step then needs a column adjusted.
    """
    rows = []
    for observation in api.search_observations(
            taxon_name=taxon_name, place_id=place_id, limit=limit,
            session=session, **search_kwargs):
        row = observation_row(observation, photo_size=photo_size)
        if row["image_url"]:
            rows.append(row)

    if not rows:
        raise ValueError("the search returned no observations with usable photos")

    destination = paths.imports_dir(project_path) / ".inat_observations.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)

    logger.info("fetched %d observations -> %s", len(rows), destination)
    return destination


def ingest_occurrences(project_path, import_csv_path=None, taxon_name=None,
                       place_id=None, limit=None, transform=None,
                       session=None, **search_kwargs):
    """
    Ingest iNaturalist observations into a project, as a full snapshot.

    A query REPLACES the project's occurrences rather than adding to them, so
    running a second query for another taxon or place doesn't accumulate the
    two. To combine several queries, run fetch_observations() for each -- it
    returns a CSV path and doesn't touch the project -- concatenate those CSVs,
    and ingest the result with import_csv_path. That keeps the combined set
    written down as one file you can re-ingest, rather than as a sequence of
    calls someone has to remember to repeat in order.

    project_path    -- project to ingest into; created lazily by the first
                      writer, so the directory needn't exist yet.
    import_csv_path -- a CSV already written by fetch_observations(). Omit to
                      search the API now.
    taxon_name      -- taxon to search under, e.g. "Odonata".
    place_id        -- iNaturalist place id to restrict to.
    limit           -- cap on observations fetched.
    transform       -- optional callable(df) -> df run after normalization.

    Returns the resulting occurrence table.
    """
    fetched = None
    if import_csv_path is None:
        fetched = fetch_observations(project_path, taxon_name=taxon_name,
                                     place_id=place_id, limit=limit,
                                     session=session, **search_kwargs)
        import_csv_path = fetched

    try:
        return core_ingest_occurrences(
            project_path,
            import_csv_path,
            datetime_cols=DATETIME_COLS,
            numeric_cols=NUMERIC_COLS,
            transform=transform,
            name_prefix=f"occurrences_inat_{taxon_name or 'query'}".replace(" ", "_"),
        )
    finally:
        # the dated archive the core ingest wrote is the durable copy
        if fetched and fetched.exists():
            fetched.unlink()
