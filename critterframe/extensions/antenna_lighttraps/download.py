"""Download Antenna detection crops into a project's image store."""

import logging

from ...download import download_images as core_download_images

logger = logging.getLogger(__name__)


def download_images(project_path, subset=None, limit=None, session=None, **kwargs):
    """Download the crop images for a project's Antenna occurrences.

    Args:
        project_path: Project to download for.
        subset: Named subset to download.
        limit: Cap on the occurrences considered.
        session: A `requests.Session` to reuse; it needn't be authenticated.
        **kwargs: Passed to `critterframe.download.download_images`.
    """
    return core_download_images(project_path, subset=subset, limit=limit, session=session, **kwargs)
