"""
Download images from the URLs in a project's ingested occurrences.

Only occurrences with no image are fetched, so this is safe to rerun: an
interrupted download resumes, and a project that gained a hundred occurrences
downloads a hundred images. A stored image is NEVER replaced -- it is the
evidence every mask and measurement was derived from, and swapping it would
leave all of them silently describing pixels that are no longer there.

Individual failures are logged and counted, never fatal.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from .project import paths, subsets as subset_selection
from .records.occurrences import ID_COL, IMAGE_URL_COL, load_occurrences
from .storage.imagestore import ImageStore

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT = (10, 60)     # connect timeout, read timeout
USER_AGENT = "critterframe-image-download/1.0"

# A polite number of concurrent connections to one host -- enough to matter
# (downloads are pure network I/O, so this is close to a free multi-x
# speedup), not aggressive enough to look like abuse to a single-host API.
# max_workers=1 reproduces the old fully-sequential behaviour, for a source
# with a strict rate limit.
DEFAULT_MAX_WORKERS = 8


def make_session():
    """A reusable HTTP session with the package's user agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _check_decodable(content, occurrence_id=None):
    """
    Confirm downloaded bytes are a decodable image, and return them unchanged.

    Validation only -- the decoded array is thrown away, since what gets stored
    is the bytes exactly as they arrived. Worth doing because a URL that 200s
    with an HTML error page is otherwise only discovered much later, by a
    segmentation run that fails on an image nobody can look at.

    content       -- raw image bytes.
    occurrence_id -- used only for error messages.
    """
    if not content:
        raise ValueError(f"empty response for {occurrence_id}")

    if cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise ValueError(f"response for {occurrence_id} isn't a decodable image")
    return content


def _download_image(url, session, occurrence_id=None, timeout=DEFAULT_TIMEOUT):
    """
    Download one image and return its encoded bytes, validated as decodable.

    url           -- image URL.
    session       -- requests.Session to issue the GET with.
    occurrence_id -- used only for error messages.
    timeout       -- (connect, read) timeout tuple.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"missing image URL for {occurrence_id}")

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return _check_decodable(response.content, occurrence_id=occurrence_id)


def _pending_occurrences(project_path, store, url_col=IMAGE_URL_COL, subset=None,
                         limit=None):
    """
    Occurrences with a URL whose image isn't in the store yet.

    Occurrences that already have an image are excluded with no way to ask
    otherwise -- see the module docstring.
    """
    # A URL column is optional in a project -- one whose images came from a
    # local folder has none at all -- so its absence is a wrong-function
    # mistake rather than a broken table, and deserves saying so. Reading it
    # off the parquet blind raises a bare ArrowInvalid naming the field but not
    # the problem.
    available = set(load_occurrences(project_path, missing_ok=True).columns)
    if url_col not in available:
        raise KeyError(
            f"no '{url_col}' column in this project's occurrences, so there "
            f"are no URLs to download from (columns: {sorted(available)}). If "
            "the images are local files, use critterframe.ingest_images() "
            "instead; if the column is named something else, pass url_col."
        )

    occurrences = subset_selection.select_occurrences(
        project_path, subset=subset, columns=[url_col])
    occurrences = occurrences.dropna(subset=[url_col])

    stored = set(store.keys())
    occurrences = occurrences[~occurrences[ID_COL].isin(stored)]

    if limit is not None:
        occurrences = occurrences.head(limit)

    return occurrences


def download_images(project_path, url_col=IMAGE_URL_COL, subset=None, limit=None,
                    batch_size=DEFAULT_BATCH_SIZE, timeout=DEFAULT_TIMEOUT,
                    session=None, max_workers=DEFAULT_MAX_WORKERS):
    """
    Download images for a project's occurrences into its image store.

    Only occurrences with no image are fetched; a stored image is never
    replaced. Bytes are stored exactly as served, and checked to be decodable on
    the way past.

    Fetches run concurrently, but only the fetch: batching and every
    store.put_many() happen on the calling thread, so nothing new touches the
    image store concurrently.

    project_path -- project whose occurrences to download for.
    url_col      -- occurrence column holding the URLs.
    subset       -- name of a subset to download, or None for all.
    limit        -- optional cap, for trying a source out.
    batch_size   -- images written to the store per LMDB transaction, flushed
                    periodically so an interruption costs at most one batch.
    timeout      -- (connect, read) timeout tuple.
    session      -- optional requests.Session to reuse.
    max_workers  -- concurrent fetches. 1 downloads strictly one at a time, e.g.
                    for a source with a strict rate limit.

    Returns a summary dict (attempted, saved, failed, failures).
    """
    paths.require_project(project_path)

    owns_session = session is None
    session = session or make_session()

    saved = 0
    failures = []
    batch = []

    try:
        with ImageStore(project_path) as store:
            pending = _pending_occurrences(project_path, store, url_col=url_col,
                                           subset=subset, limit=limit)
            attempted = len(pending)
            logger.info("%d image(s) pending download", attempted)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(_download_image, getattr(row, url_col), session,
                                    occurrence_id=getattr(row, ID_COL), timeout=timeout):
                        (getattr(row, ID_COL), getattr(row, url_col))
                    for row in pending.itertuples(index=False)
                }

                for future in as_completed(future_to_row):
                    occurrence_id, url = future_to_row[future]

                    try:
                        batch.append((occurrence_id, future.result()))
                        if len(batch) >= batch_size:
                            store.put_many(batch)
                            saved += len(batch)
                            batch.clear()
                            logger.info("saved %d/%d", saved, attempted)

                    except Exception as exc:
                        logger.warning("download failed for %s: %s", occurrence_id, exc)
                        failures.append({"occurrence_id": occurrence_id, "url": url,
                                         "error": str(exc)})

            if batch:
                store.put_many(batch)
                saved += len(batch)
    finally:
        if owns_session:
            session.close()

    logger.info("image download complete: attempted=%d saved=%d failed=%d",
                attempted, saved, len(failures))
    return {"attempted": attempted, "saved": saved, "failed": len(failures),
            "failures": failures}
