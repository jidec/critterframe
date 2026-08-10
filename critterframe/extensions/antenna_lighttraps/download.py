"""
Download Antenna detection crops into a project's image store.

Thin on purpose. Antenna serves its crops as ordinary public image URLs, so the
core downloader already does the work -- what belongs here is only the fact
that Antenna's crops are the analysis images (not thumbnails of something
larger), and that they're immutable once cut, so a stored one never needs
re-fetching.

The core download_images() is what to call in a general pipeline; this wrapper
exists so an Antenna pipeline script reads consistently with its ingest step,
and as the place any Antenna-specific fetching would go if the API ever needs
authenticated image requests or a choice between crop sizes.
"""

import logging

from ...download import download_images as core_download_images

logger = logging.getLogger(__name__)


def download_images(project_path, subset=None, limit=None, session=None, **kwargs):
    """
    Download the crop images for a project's Antenna occurrences.

    project_path -- project whose occurrences to download for.
    subset       -- name of a subset to download, or None for all.
    limit        -- optional cap.
    session      -- optional requests.Session to reuse. An authenticated
                    Antenna session works but isn't required: crop URLs are
                    served from public object storage, not the API.

    Everything else is passed through to critterframe.download.download_images.
    """
    return core_download_images(project_path, subset=subset, limit=limit,
                                session=session, **kwargs)
