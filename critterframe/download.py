"""Download images from the URLs in a project's occurrences into its image store."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from . import drivers
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
DEFAULT_TIMEOUT = (10, 60)  # connect timeout, read timeout
USER_AGENT = "critterframe-image-download/1.0"

# A polite number of concurrent connections to one host -- enough to matter
# (downloads are pure network I/O, so this is close to a free multi-x
# speedup), not aggressive enough to look like abuse to a single-host API.
# max_workers=1 reproduces the old fully-sequential behaviour, for a source
# with a strict rate limit.
DEFAULT_MAX_WORKERS = 8


def make_session(user_agent=USER_AGENT, min_interval=None):
    """Return a reusable HTTP session with a user agent and an optional rate limit.

    Args:
        user_agent: What to identify as.
        min_interval: Minimum seconds between requests, enforced across threads.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    if min_interval:
        _pace(session, min_interval)
    return session


def _pace(session, min_interval):
    """Wrap `session.get` so no two calls, on any thread, start closer together than `min_interval`."""
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
    """Confirm downloaded bytes decode as an image, and return them unchanged.

    Args:
        content: Raw image bytes.
        occurrence_id: For the error message.
    """
    if not content:
        raise ValueError(f"empty response for {occurrence_id}")

    if cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise ValueError(f"response for {occurrence_id} isn't a decodable image")
    return content


def _download_image(url, session, occurrence_id=None, timeout=DEFAULT_TIMEOUT):
    """Download one image and return its encoded bytes, checked to be decodable.

    Args:
        url: Image URL.
        session: The `requests.Session` to use.
        occurrence_id: For the error message.
        timeout: `(connect, read)` timeout.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"missing image URL for {occurrence_id}")

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return _check_decodable(response.content, occurrence_id=occurrence_id)


def _thumbnail(content):
    """Return a downloaded image decoded for a grid cell, captioned with its size, or None."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    annotate(image, f"{width}x{height} {len(content) / 1024:.0f} KB")
    return image


def _url_context_hash(url):
    """Return the identity of one download attempt, so a changed URL is retried."""
    return hash_spec({"url": url})


def _pending_occurrences(
    project_path, store, url_col=IMAGE_URL_COL, subset=None, limit=None, max_new=None, retry_failed=False
):
    """Return the occurrences with a URL whose image is not in the store.

    Excludes URLs that already failed unless `retry_failed`. `limit` narrows the occurrences
    considered; `max_new` caps what is left.

    Returns:
        `(pending_df, previously_failed_count)`.
    """
    # A URL column is optional in a project -- one whose images came from a
    # local folder has none at all -- so its absence is a wrong-function
    # mistake rather than a broken table, and deserves saying so. Checked
    # against the parquet's schema rather than by loading the whole table,
    # which on a 500,000-row project is a lot of work to answer one question.
    require_columns(
        project_path,
        url_col,
        "nothing to download from -- if the images are local files "
        "use critterframe.ingest_images(), and if the column is "
        "named something else pass url_col",
    )

    occurrences = subset_selection.select_occurrences(
        project_path, subset=subset, limit=limit, columns=[url_col]
    )
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


def download_images(
    project_path,
    url_col=IMAGE_URL_COL,
    subset=None,
    limit=None,
    max_new=None,
    batch_size=DEFAULT_BATCH_SIZE,
    timeout=DEFAULT_TIMEOUT,
    session=None,
    max_workers=DEFAULT_MAX_WORKERS,
    retry_failed=False,
    visualize=True,
    visualize_every=None,
):
    """Download images for a project's occurrences into its image store.

    Only occurrences with no image are fetched, and a stored image is never replaced.
    Bytes are stored as served, after checking they decode.

    Args:
        project_path: Project to download for.
        url_col: Occurrence column holding the URLs.
        subset: Named subset to download.
        limit: Cap on the occurrences considered, before stored or failed ones are excluded.
        max_new: Cap on how many are fetched in this call.
        batch_size: Images written per store transaction; an interruption loses at most one batch.
        timeout: `(connect, read)` timeout.
        session: A `requests.Session` to reuse.
        max_workers: Concurrent fetches; 1 downloads one at a time.
        retry_failed: Attempt URLs that failed on an earlier call. A changed URL is retried
            regardless.
        visualize: True, an int, or ids: a pipeline grid of thumbnails of what was saved.
        visualize_every: Also write a thumbnail grid every N downloads.

    Returns:
        The `drivers.Tally.summary` dict plus `previously_failed` and `elapsed_s`.
    """
    paths.require_project(project_path)

    owns_session = session is None
    session = session or make_session()

    tally = drivers.Tally()
    batch = []

    try:
        with ImageStore(project_path) as store:
            pending, previously_failed = _pending_occurrences(
                project_path,
                store,
                url_col=url_col,
                subset=subset,
                limit=limit,
                max_new=max_new,
                retry_failed=retry_failed,
            )
            tally.attempted = len(pending)
            logger.info(
                "%d image(s) pending download (%d previously failed, skipped)",
                tally.attempted,
                previously_failed,
            )

            ordered_ids = [str(occurrence_id) for occurrence_id in pending[ID_COL]]
            identity = {
                "kind": "download",
                "url_col": url_col,
                "subset": subset,
                "attempted": ids_record(ordered_ids),
            }
            report = pipeline_visualization.open_report(
                project_path,
                "download",
                hash_spec(identity),
                visualize=visualize,
                visualize_every=visualize_every,
                identity=identity,
            ).begin(ordered_ids)
            planned = report.planned()

            # Fetches finish out of order, but checkpoint windows are positional,
            # so each item is handed to the report when its turn comes. Only the
            # bytes a grid will actually show are held until then.
            finished = {}
            next_position = 0

            def advance():
                nonlocal next_position
                while next_position < len(ordered_ids) and ordered_ids[next_position] in finished:
                    occurrence_id = ordered_ids[next_position]
                    content = finished.pop(occurrence_id)
                    if content is not None:
                        thumbnail = _thumbnail(content)
                        if thumbnail is not None:
                            report.panel(occurrence_id, "downloaded", thumbnail)
                    report.done(occurrence_id)
                    next_position += 1

            progress = drivers.Progress(len(ordered_ids), "download_images", tallies=[tally], log=logger.info)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(
                        _download_image,
                        getattr(row, url_col),
                        session,
                        occurrence_id=getattr(row, ID_COL),
                        timeout=timeout,
                    ): (getattr(row, ID_COL), getattr(row, url_col))
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
                                project_path,
                                "download",
                                keys=[(oid, failure_records.NO_PART) for oid, _ in batch],
                            )
                            tally.processed += len(batch)
                            batch.clear()

                    except Exception as exc:
                        logger.warning("download failed for %s: %s", occurrence_id, exc)
                        tally.record_failure(occurrence_id, exc, url=url)
                        report.failure(occurrence_id, f"{url}: {exc}")
                        finished[str(occurrence_id)] = None

                    progress.step()
                    advance()

            if batch:
                store.put_many(batch)
                failure_records.clear_failures(
                    project_path, "download", keys=[(oid, failure_records.NO_PART) for oid, _ in batch]
                )
                tally.processed += len(batch)

            if tally.failures:
                failure_records.record_failures(
                    project_path,
                    "download",
                    [
                        {
                            "occurrence_id": failure["occurrence_id"],
                            "context_hash": _url_context_hash(failure["url"]),
                            "error": failure["error"],
                        }
                        for failure in tally.failures
                    ],
                )
            report.close()
            elapsed = progress.finish()
    finally:
        if owns_session:
            session.close()

    logger.info(
        "image download complete: attempted=%d saved=%d failed=%d",
        tally.attempted,
        tally.processed,
        tally.failed,
    )
    return tally.summary(previously_failed=previously_failed, elapsed_s=elapsed)
