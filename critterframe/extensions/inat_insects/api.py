"""
iNaturalist API client.

Reads observations from the public iNaturalist API v1
(https://api.inaturalist.org/v1). No authentication is needed for public
observation search, which is all this does.

Two things about the API shape the code here:

  Pagination is by id, not by page. The `page` parameter stops working past
  10,000 results, so anything larger has to walk forward with `id_above`
  ordered by id ascending -- which is also more robust to observations being
  added while you page, since a page-numbered walk would shift under you.

  Rate limits are real and enforced. The published guidance is at most one
  request per second and to keep daily volume modest, so requests are spaced
  deliberately. A pull of tens of thousands of observations takes as long as
  it takes; going faster gets an IP blocked, which costs far more time than
  the waiting did.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE = "https://api.inaturalist.org/v1"

# Seconds between requests. The published limit is 1/second; 1.1 leaves margin
# for clock differences rather than sitting exactly on the boundary.
REQUEST_INTERVAL = 1.1

# Results per request. 200 is the API's maximum and the fewest round trips.
PER_PAGE = 200

USER_AGENT = "critterframe/0.1 (research image analysis)"

# Photo size to request. iNaturalist serves several; "medium" (~500px) is
# usually too small to measure an organism from, and "original" is large and
# slow -- "large" (~1024px) is the compromise, and it's what the default
# segmentation resolution expects anyway.
PHOTO_SIZE = "large"


def make_session():
    """A requests.Session identifying itself, as the API asks callers to do."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def search_observations(taxon_name=None, place_id=None, quality_grade="research",
                        photos=True, licensed=True, limit=None, session=None,
                        per_page=PER_PAGE, extra_params=None):
    """
    Walk observations matching a query, yielding one observation dict at a time.

    A generator rather than a list: a broad query can return hundreds of
    thousands of observations, and a caller filtering as it goes shouldn't have
    to hold them all first.

    taxon_name    -- taxon to search under, e.g. "Odonata". Matched by name,
                     so it covers every descendant.
    place_id      -- iNaturalist place id to restrict to. Names aren't accepted
                     by the API; look the id up once on the website.
    quality_grade -- "research" (default) restricts to community-verified
                     identifications. Worth keeping: an unverified
                     identification makes a per-species group metric
                     meaningless, since the groups themselves would be wrong.
    photos        -- require at least one photo. An observation without one
                     can't become a CritterFrame occurrence.
    licensed      -- require a license permitting reuse. Downloading and
                     analyzing all-rights-reserved photos in bulk isn't
                     something to do by default.
    limit         -- stop after this many observations.
    session       -- requests.Session to reuse; one is made if omitted.
    per_page      -- results per request.
    extra_params  -- any other API parameters, passed through untouched.
    """
    session = session or make_session()

    params = {
        "per_page": per_page,
        "order_by": "id",
        "order": "asc",
        "photos": str(bool(photos)).lower(),
    }
    if taxon_name:
        params["taxon_name"] = taxon_name
    if place_id:
        params["place_id"] = place_id
    if quality_grade:
        params["quality_grade"] = quality_grade
    if licensed:
        params["photo_license"] = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
    if extra_params:
        params.update(extra_params)

    yielded = 0
    id_above = 0

    while True:
        response = session.get(f"{BASE}/observations",
                               params={**params, "id_above": id_above},
                               timeout=60)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            break

        for observation in results:
            yield observation
            yielded += 1
            if limit is not None and yielded >= limit:
                logger.info("stopped at limit of %d observations", limit)
                return

        id_above = results[-1]["id"]
        logger.info("fetched %d observations (through id %s)", yielded, id_above)
        time.sleep(REQUEST_INTERVAL)


def photo_url(observation, size=PHOTO_SIZE, index=0):
    """
    The URL of one of an observation's photos at the requested size.

    iNaturalist gives a "square" (75px) URL in the observation payload and
    expects callers to substitute the size they want into the filename, which
    is what this does. Returns None if the observation has no photo at `index`.

    index -- which photo. 0 (the first) by default, since the first is the one
             an observer chose as representative -- and because one occurrence
             is one image, so taking several photos of one observation would
             create several occurrences of the same organism.
    """
    photos = observation.get("photos") or []
    if index >= len(photos):
        return None

    url = photos[index].get("url")
    if not url:
        return None
    return url.replace("square", size)


def taxon_name(observation, rank="species"):
    """
    The observation's identified taxon name at a given rank, or None.

    Falls back to whatever the observation IS identified to when it isn't
    identified to `rank` -- an observation determined only to genus still has a
    usable group for a group metric, and dropping it would silently bias the
    reference population toward well-studied taxa.
    """
    taxon = observation.get("taxon") or {}
    if not taxon:
        return None

    if taxon.get("rank") == rank:
        return taxon.get("name")

    for ancestor in reversed(taxon.get("ancestors") or []):
        if ancestor.get("rank") == rank:
            return ancestor.get("name")

    return taxon.get("name")
