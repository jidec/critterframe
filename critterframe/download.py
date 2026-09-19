"""
Download images from the URLs in a project's ingested occurrences.

Only occurrences with no image are fetched, so this is safe to rerun: an
interrupted download resumes, and a project that gained a hundred occurrences
downloads a hundred images. A stored image is NEVER replaced -- it is the
evidence every mask and measurement was derived from, and swapping it would
leave all of them silently describing pixels that are no longer there.

Individual failures are logged, counted, and recorded in records.failures
against the URL that failed, so a rerun does not re-fetch a dead URL --
retry_failed=True asks anyway, and a URL that changes (a corrected re-ingest)
is retried automatically with no flag needed.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from . import segments as segment_iteration
from . import selectionhelpers
from .project import paths, subsets as subset_selection
from .records import failures as failure_records
from .records.occurrences import ID_COL, IMAGE_URL_COL, ids_record, require_columns
from .recipes import hash_spec
from .storage.imagestore import ImageStore
from .visualization import pipeline as pipeline_visualization
from .visualization.panels import annotate

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


def make_session(user_agent=USER_AGENT, min_interval=None):
    """
    A reusable HTTP session with a user agent and, optionally, a rate limit.

    - `user_agent` -- what to identify as. A source with its own etiquette
      passes its own.
    - `min_interval` -- minimum seconds between requests, enforced across
      THREADS: `download_images` fetches concurrently, so an unsynchronized
      timestamp would let every worker fire at once and pace nothing.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    if min_interval:
        _pace(session, min_interval)
    return session


def _pace(session, min_interval):
    """Wrap session.get so no two calls, on any thread, start closer than min_interval."""
    original_get = session.get
    lock = threading.Lock()
    state = {"next_allowed": 0.0}

    def paced_get(*args, **kwargs):
        with lock:
            wait = state["next_allowed"] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            state["next_allowed"] = time.monotonic() + min_interval
        return original_get(*args, **kwargs)

    session.get = paced_get
    return session


def _check_decodable(content, occurrence_id=None):
    """
    Confirm downloaded bytes are a decodable image, and return them unchanged.

    Validation only -- the decoded array is thrown away, since what gets stored
    is the bytes exactly as they arrived. Worth doing because a URL that 200s
    with an HTML error page is otherwise only discovered much later, by a
    segmentation run that fails on an image nobody can look at.

    - `content` -- raw image bytes.
    - `occurrence_id` -- used only for error messages.
    """
    if not content:
        raise ValueError(f"empty response for {occurrence_id}")

    if cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise ValueError(f"response for {occurrence_id} isn't a decodable image")
    return content


def _download_image(url, session, occurrence_id=None, timeout=DEFAULT_TIMEOUT):
    """
    Download one image and return its encoded bytes, validated as decodable.

    - `url` -- image URL.
    - `session` -- requests.Session to issue the GET with.
    - `occurrence_id` -- used only for error messages.
    - `timeout` -- (connect, read) timeout tuple.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"missing image URL for {occurrence_id}")

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return _check_decodable(response.content, occurrence_id=occurrence_id)


def _thumbnail(content):
    """A downloaded image decoded for a grid cell, captioned with its size, or None if it won't decode."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    annotate(image, f"{width}x{height} {len(content) / 1024:.0f} KB")
    return image


def _url_context_hash(url):
    """The identity of one download attempt: what failed_keys() scopes a
    recorded failure to, so a corrected URL is retried with no flag needed."""
    return hash_spec({"url": url})


def _pending_occurrences(project_path, store, url_col=IMAGE_URL_COL, subset=None,
                         limit=None, max_new=None, retry_failed=False):
    """
    Occurrences with a URL whose image isn't in the store yet and, unless
    retry_failed, whose URL hasn't already failed.

    Occurrences that already have an image are excluded with no way to ask
    otherwise -- see the module docstring. A previously-failed URL is skipped
    the same way, but retry_failed=True or a changed URL both let it through.

    `limit` narrows the occurrences considered, before any of that; `max_new`
    caps what is left after it. See download_images.

    Returns (pending_df, previously_failed_count).
    """
    # A URL column is optional in a project -- one whose images came from a
    # local folder has none at all -- so its absence is a wrong-function
    # mistake rather than a broken table, and deserves saying so. Checked
    # against the parquet's schema rather than by loading the whole table,
    # which on a 500,000-row project is a lot of work to answer one question.
    require_columns(project_path, url_col,
                    "nothing to download from -- if the images are local files "
                    "use critterframe.ingest_images(), and if the column is "
                    "named something else pass url_col")

    occurrences = subset_selection.select_occurrences(
        project_path, subset=subset, limit=limit, columns=[url_col])
    occurrences = occurrences.dropna(subset=[url_col])

    keep = selectionhelpers.exclude_present(occurrences[ID_COL], stored=store.keys())
    occurrences = occurrences[occurrences[ID_COL].isin(keep)]

    previously_failed = 0
    if not retry_failed and len(occurrences):
        context_hashes = {
            (occurrence_id, failure_records.NO_PART): _url_context_hash(url)
            for occurrence_id, url in zip(occurrences[ID_COL], occurrences[url_col])
        }
        failed = failure_records.failed_keys(project_path, "download", context_hashes)
        if failed:
            failed_ids = {occurrence_id for occurrence_id, _ in failed}
            previously_failed = len(failed_ids)
            occurrences = occurrences[~occurrences[ID_COL].isin(failed_ids)]

    if max_new is not None:
        occurrences = occurrences.head(max_new)

    return occurrences, previously_failed


def download_images(project_path, url_col=IMAGE_URL_COL, subset=None, limit=None,
                    max_new=None, batch_size=DEFAULT_BATCH_SIZE,
                    timeout=DEFAULT_TIMEOUT, session=None,
                    max_workers=DEFAULT_MAX_WORKERS, retry_failed=False,
                    visualize=True, visualize_every=None):
    """
    Download images for a project's occurrences into its image store.

    Only occurrences with no image are fetched; a stored image is never
    replaced. Bytes are stored exactly as served, and checked to be decodable on
    the way past. A URL that has already failed is skipped on a rerun too,
    unless retry_failed asks for it or the URL itself has changed.

    Fetches run concurrently, but only the fetch: batching and every
    store.put_many() happen on the calling thread, so nothing new touches the
    image store concurrently.

    - `project_path` -- project whose occurrences to download for.
    - `url_col` -- occurrence column holding the URLs.
    - `subset` -- name of a subset to download, or None for all.
    - `limit` -- cap on the occurrences CONSIDERED, applied before anything
      already stored or already failed is excluded -- the same meaning every
      driver's `limit` has. `limit=10` against ten already-downloaded
      occurrences therefore downloads nothing.
    - `max_new` -- cap on what is actually FETCHED, applied after those
      exclusions. This is the "try this source on ten images" argument, and
      the one that downloads ten more every time it is run.
    - `batch_size` -- images written to the store per LMDB transaction, flushed
      periodically so an interruption costs at most one batch.
    - `timeout` -- (connect, read) timeout tuple.
    - `session` -- optional `requests.Session` to reuse.
    - `max_workers` -- concurrent fetches. 1 downloads strictly one at a time,
      e.g. for a source with a strict rate limit.
    - `retry_failed` -- attempt occurrences whose URL already failed on a
      previous call. False (the default) leaves them recorded as failed.
    - `visualize` -- True (default), an int, or ids: a pipeline grid of
      thumbnails of what was saved, with every failure listed in its
      sidecar. False writes nothing.
    - `visualize_every` -- also write a thumbnail grid every N downloads,
      sampled from that stretch only.

    Returns a summary dict (see `segments.Tally.summary`) plus
    `previously_failed`: URLs left alone because they failed before.
    """
    paths.require_project(project_path)

    owns_session = session is None
    session = session or make_session()

    tally = segment_iteration.Tally()
    batch = []

    try:
        with ImageStore(project_path) as store:
            pending, previously_failed = _pending_occurrences(
                project_path, store, url_col=url_col, subset=subset,
                limit=limit, max_new=max_new, retry_failed=retry_failed)
            tally.attempted = len(pending)
            logger.info("%d image(s) pending download (%d previously failed, skipped)",
                        tally.attempted, previously_failed)

            ordered_ids = [str(occurrence_id) for occurrence_id in pending[ID_COL]]
            identity = {"kind": "download", "url_col": url_col, "subset": subset,
                        "attempted": ids_record(ordered_ids)}
            report = pipeline_visualization.open_report(
                project_path, "download", hash_spec(identity), visualize=visualize,
                visualize_every=visualize_every, identity=identity).begin(ordered_ids)
            planned = report.planned()

            # Fetches finish out of order, but checkpoint windows are positional,
            # so each item is handed to the report when its turn comes. Only the
            # bytes a grid will actually show are held until then.
            finished = {}
            next_position = 0

            def advance():
                nonlocal next_position
                while next_position < len(ordered_ids) and                         ordered_ids[next_position] in finished:
                    occurrence_id = ordered_ids[next_position]
                    content = finished.pop(occurrence_id)
                    if content is not None:
                        thumbnail = _thumbnail(content)
                        if thumbnail is not None:
                            report.panel(occurrence_id, "downloaded", thumbnail)
                    report.done(occurrence_id)
                    next_position += 1

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
                        content = future.result()
                        batch.append((occurrence_id, content))
                        wanted = planned is None or str(occurrence_id) in planned
                        finished[str(occurrence_id)] = content if wanted else None
                        if len(batch) >= batch_size:
                            store.put_many(batch)
                            failure_records.clear_failures(
                                project_path, "download",
                                keys=[(oid, failure_records.NO_PART) for oid, _ in batch])
                            tally.processed += len(batch)
                            batch.clear()
                            logger.info("saved %d/%d", tally.processed, tally.attempted)

                    except Exception as exc:
                        logger.warning("download failed for %s: %s", occurrence_id, exc)
                        tally.record_failure(occurrence_id, exc, url=url)
                        report.failure(occurrence_id, f"{url}: {exc}")
                        finished[str(occurrence_id)] = None

                    advance()

            if batch:
                store.put_many(batch)
                failure_records.clear_failures(
                    project_path, "download",
                    keys=[(oid, failure_records.NO_PART) for oid, _ in batch])
                tally.processed += len(batch)

            if tally.failures:
                failure_records.record_failures(project_path, "download", [
                    {"occurrence_id": failure["occurrence_id"],
                     "context_hash": _url_context_hash(failure["url"]),
                     "error": failure["error"]}
                    for failure in tally.failures
                ])
            report.close()
    finally:
        if owns_session:
            session.close()

    logger.info("image download complete: attempted=%d saved=%d failed=%d",
                tally.attempted, tally.processed, tally.failed)
    return tally.summary(previously_failed=previously_failed)
