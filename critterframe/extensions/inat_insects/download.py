"""
Download iNaturalist photos into a project's image store.

The core downloader does the fetching; what this adds is pacing. iNaturalist
photos are served from a CDN rather than the rate-limited API, but a project
pulling tens of thousands of images is still a burst of traffic aimed at a
volunteer-funded service, and the polite default is to not send it as fast as
the network allows.

That's the only real difference, and it's a deliberate one: the right speed to
download someone else's images is slower than the maximum.
"""

import logging
import time

from ...download import DEFAULT_TIMEOUT
from ...download import download_images as core_download_images
from ...download import make_session

logger = logging.getLogger(__name__)

# Seconds between photo requests. Not an enforced limit -- a courtesy one.
DOWNLOAD_INTERVAL = 0.2

USER_AGENT = "critterframe/0.1 (research image analysis)"


def make_paced_session(interval=DOWNLOAD_INTERVAL):
    """
    A session that waits `interval` seconds between requests.

    Implemented by wrapping get() rather than with an adapter, so it works
    unchanged with the core downloader, which only ever calls get().
    """
    session = make_session()
    session.headers.update({"User-Agent": USER_AGENT})

    original_get = session.get
    state = {"last": 0.0}

    def paced_get(*args, **kwargs):
        elapsed = time.perf_counter() - state["last"]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        state["last"] = time.perf_counter()
        return original_get(*args, **kwargs)

    session.get = paced_get
    return session


def download_images(project_path, subset=None, limit=None,
                    interval=DOWNLOAD_INTERVAL, timeout=DEFAULT_TIMEOUT, **kwargs):
    """
    Download photos for a project's iNaturalist occurrences, paced.

    project_path -- project whose occurrences to download for.
    subset       -- name of a subset to download, or None for all.
    limit        -- optional cap.
    interval     -- minimum seconds between requests.

    Everything else is passed through to critterframe.download.download_images,
    including its skip-what's-already-stored behaviour -- so an interrupted
    download of a large project resumes rather than starting over, which
    matters much more here than it does for a fast source.
    """
    session = make_paced_session(interval=interval)
    try:
        return core_download_images(project_path, subset=subset, limit=limit,
                                    session=session, timeout=timeout, **kwargs)
    finally:
        session.close()
