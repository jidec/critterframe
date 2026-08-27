"""
Download images from the URLs in a project's ingested occurrences.

Generic on purpose: it reads one column of URLs and writes one image per
occurrence into the image store, and knows nothing about who served them.
Source-specific downloading -- an API's rate limits, its authentication, its
choice of which of several photo sizes to fetch -- lives in extensions/ and
normally ends up calling this with a URL column it prepared.

Only occurrences missing an image are fetched, so this is safe to rerun: an
interrupted download resumes where it stopped, and a project that gained a
hundred new occurrences downloads a hundred images rather than all of them
again.

ONCE AN IMAGE IS STORED FOR AN OCCURRENCE, NOTHING HERE REPLACES IT. There is
no override, deliberately. An analysis image is the evidence every mask and
every measurement in the project was derived from, and swapping it out leaves
all of that in place, silently describing pixels that are no longer there --
with no record that anything moved, because the store keeps one image per
occurrence and no history. A source that genuinely reissued its files is a new
import: ingest it as its own project, or delete the specific keys from the store
yourself, having decided what that invalidates.

Individual failures are logged and counted, never fatal. A dead URL in a
50,000-row export shouldn't cost you the other 49,999.
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
    Confirm downloaded bytes are a decodable image, and return them UNCHANGED.

    Validation only -- the decoded array is thrown away deliberately. What gets
    stored is the bytes exactly as they arrived off the wire (see
    storage.imagestore); decoding here and storing the array instead would
    re-encode every image and lose whatever the original format was carrying.

    Worth doing even so: a URL that 200s with an HTML error page, or a truncated
    response, is otherwise only discovered much later, by a segmentation run
    that fails on an image nobody can look at.

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
    Download one image and return its encoded BYTES, validated as decodable.

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
    otherwise -- see the module docstring for why a stored image is never
    replaced from here.
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

    Only occurrences with no image yet are fetched, always: an occurrence that
    already has one is left alone, and there is no parameter to change that (see
    the module docstring).

    Images are stored as the exact bytes served, whatever format that is -- no
    decode, no re-encode (see storage.imagestore). Each response is checked to
    be a decodable image on the way past, so a URL that 200s with an error page
    fails here rather than during a segmentation run days later.

    Fetches run concurrently across a thread pool -- this is pure network I/O,
    so threads (not processes) are the right tool, and a single shared
    requests.Session is safe to use from many threads at once (its connection
    pool is designed for exactly this). Only the fetch itself is threaded:
    batching and the actual store.put_many() write always happen on the calling
    thread, one at a time, so nothing new touches the image store concurrently.

    project_path -- project whose occurrences to download for.
    url_col      -- occurrence column holding the URLs.
    subset       -- name of a subset to download, or None for all.
    limit        -- optional cap, for trying a source out before committing.
    batch_size   -- images written to the store per transaction. Batched
                    because one LMDB transaction per image is markedly slower;
                    flushed periodically rather than at the end so an
                    interruption costs at most one batch.
    timeout      -- (connect, read) timeout tuple.
    session      -- optional requests.Session to reuse; one is created and
                    closed if omitted.
    max_workers  -- concurrent fetches. Set to 1 to download strictly one at a
                    time, e.g. for a source with a strict per-second rate limit
                    (rate limiting itself is an extension's concern, not this
                    module's -- see the module docstring).

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
