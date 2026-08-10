"""
Antenna API connection and authentication.

Antenna (https://api.antenna.insectai.org/api/v2) is an external
insect-monitoring platform. It runs detection and classification over light-trap
sheet images and exposes the results as per-detection crops with metadata --
which is to say it has already done the "one organism per image" separation
CritterFrame requires upstream of ingest.

This module is the transport layer only: log in, ask for an export, wait for it,
stream it to disk, page through a list endpoint. Turning what comes back into
occurrences is ingest.py's job; turning captures into a calibration is
calibrations/scale.py's.

Credentials come from the environment (ANTENNA_EMAIL, ANTENNA_PW), read from a
.env file locally via python-dotenv. They're environment rather than project
configuration deliberately: a project directory is data and is copied,
archived, and shared, and a password has no business travelling with it.
"""

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()   # reads .env into the environment

logger = logging.getLogger(__name__)

BASE = os.environ.get("ANTENNA_BASE", "https://api.antenna.insectai.org/api/v2")

OCCURRENCES_FORMAT = "occurrences_simple_csv"
CLASSIFICATIONS_FORMAT = "classifications_simple_csv"

# Results per page when walking a list endpoint. ADVISORY ONLY -- the captures
# endpoint returns 20 whatever you ask for (measured: 1, 50, 200 and 500 all
# come back as 20). Still sent, in case another endpoint honours it, but never
# assume a walk is short because this is large: 3,451 captures is 173 requests.
PAGE_SIZE = 200

# How often a long walk says something. Every page would be noise; never would
# leave a multi-minute read looking like a hang, which is exactly what happened.
PROGRESS_EVERY = 10


def project_id(default=None):
    """
    The Antenna project id to work with, from ANTENNA_PROJECT_ID.

    default -- returned when the variable isn't set; raises if it's None, since
              silently exporting the wrong project is worse than failing.
    """
    value = os.environ.get("ANTENNA_PROJECT_ID")
    if value is not None:
        return int(value)
    if default is not None:
        return int(default)
    raise RuntimeError(
        "no Antenna project id -- set ANTENNA_PROJECT_ID in the environment "
        "or pass project_id explicitly"
    )


def get_session():
    """
    An authenticated requests.Session.

    Reads ANTENNA_EMAIL and ANTENNA_PW from the environment, logs in, and
    attaches the returned token to the session.
    """
    email = os.environ.get("ANTENNA_EMAIL")
    password = os.environ.get("ANTENNA_PW")
    if not (email and password):
        raise RuntimeError(
            "missing credentials -- set ANTENNA_EMAIL and ANTENNA_PW in the "
            "environment (a .env file for local work, or an EnvironmentFile on "
            "a server)"
        )

    session = requests.Session()
    response = session.post(f"{BASE}/auth/token/login/",
                            json={"email": email, "password": password},
                            timeout=30)
    response.raise_for_status()
    session.headers["Authorization"] = f"Token {response.json()['auth_token']}"
    logger.info("authenticated with Antenna as %s", email)
    return session


def paginate(session, path, params=None, page_size=PAGE_SIZE, timeout=60):
    """
    Yield every result across the pages of a list endpoint.

    Antenna's list endpoints are DRF-paginated: each response carries `results`
    and an absolute `next` url. Following `next` rather than counting pages
    ourselves means the server decides where the boundaries are, so a page size
    it silently caps, or a record inserted mid-walk, doesn't cost us rows.

    Reports progress as it goes. A whole-project walk is hundreds of requests
    and several minutes, and a silent several minutes is indistinguishable from
    a hang -- the first page's `count` is logged immediately so the size of the
    job is known before committing to it.

    session -- authenticated session from get_session().
    path    -- endpoint path relative to BASE, e.g. "captures/".
    params  -- query parameters for the FIRST request; `next` already encodes
               them for the rest.
    """
    url = f"{BASE}/{path}"
    params = dict(params or {}, page_size=page_size)
    started = time.time()
    pages = 0
    seen = 0

    while url:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        page = response.json()

        pages += 1
        if pages == 1 and page.get("count") is not None:
            logger.info("%s: walking %s record(s)", path, page["count"])
        elif pages % PROGRESS_EVERY == 0:
            logger.info("%s: %d record(s) after %d pages, %.0fs elapsed",
                        path, seen, pages, time.time() - started)

        seen += len(page["results"])
        yield from page["results"]

        url = page.get("next")
        params = None

    logger.info("%s: %d record(s) in %d page(s), %.0fs",
                path, seen, pages, time.time() - started)


# Query parameter that narrows captures to one event. The name matters more
# than it looks: the endpoint takes `event`, and an unrecognised filter is
# DROPPED SILENTLY rather than rejected -- `?event_id=12835` returns the whole
# project's 3,451 captures with a 200, not an error. So a plausible-looking
# capture from the wrong night is what a typo here buys you, which is why
# callers verify the event on what comes back (see calibrations/scale.py).
EVENT_PARAM = "event"


def fetch_captures(session, project=None, event=None, page_size=PAGE_SIZE):
    """
    Captures (light-trap sheet images) in a project, as raw API records.

    A capture is the whole sheet a night's detections were cut out of, which is
    the image a reference card is visible in -- the occurrence export only
    carries the crops. Each record carries its event inline and a PRESIGNED url,
    so a caller needs neither a bucket nor credentials to fetch the image
    itself; see calibrations/scale.py, the reason this exists.

    project -- Antenna project id; from the environment if omitted. Required by
               the endpoint, which 400s without it.
    event   -- narrow to one event's captures. Worth doing whenever you can: the
               whole-project walk is 3,451 records over 173 requests for this
               project, and one event is a single request. Filtering happens
               server-side under EVENT_PARAM -- read the note there before
               changing the name.
    """
    params = {"project_id": project_id(project)}
    if event is not None:
        params[EVENT_PARAM] = event

    return paginate(session, "captures/", params, page_size=page_size)


def request_export(session, fmt=OCCURRENCES_FORMAT, project=None, filters=None):
    """
    Request a new export and return its id.

    Every Antenna export is a FULL project snapshot rather than a delta, which
    is why ingest replaces the occurrence table wholesale rather than merging.

    session -- authenticated session from get_session().
    fmt     -- registered export format, e.g. OCCURRENCES_FORMAT.
    project -- Antenna project id; taken from the environment if omitted.
    filters -- optional server-side filter dict. Only collection_id is
               supported server-side at present; date and deployment filtering
               are not, so narrowing by those has to happen after ingest.
    """
    project = project_id(project)
    payload = {"project": project, "format": fmt}
    if filters:
        payload["filters"] = filters

    response = session.post(f"{BASE}/exports/", json=payload, timeout=30)
    response.raise_for_status()
    export_id = response.json()["id"]

    logger.info("requested %s export %s for project %s", fmt, export_id, project)
    return export_id


def poll_export(session, export_id, interval=5, timeout=1800):
    """
    Poll until an export is ready; returns its metadata dict, including
    file_url.

    export_id -- id from request_export().
    interval  -- seconds between polls.
    timeout   -- give up and raise after this many seconds. Exports of a large
                 project take a while; raise this rather than lowering it if
                 you start seeing timeouts, since a timed-out poll doesn't
                 cancel the export -- it can be fetched later by id.
    """
    started = time.time()
    while time.time() - started < timeout:
        response = session.get(f"{BASE}/exports/{export_id}/", timeout=30)
        response.raise_for_status()
        export = response.json()

        if export.get("file_url"):
            return export

        logger.info("export %s progress: %s", export_id,
                    (export.get("job") or {}).get("progress"))
        time.sleep(interval)

    raise TimeoutError(f"export {export_id} not ready after {timeout}s")


def download_export(session, export, dest):
    """
    Stream a finished export to disk and return the path.

    session -- authenticated session.
    export  -- export metadata dict from poll_export().
    dest    -- local path to write to.

    To fetch a previously-completed export without polling:
        export = session.get(f"{BASE}/exports/{export_id}/").json()
        download_export(session, export, dest)
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with session.get(export["file_url"], stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)

    logger.info("downloaded export -> %s (%s records)", dest,
                export.get("record_count"))
    return dest


def fetch_export(session, dest, fmt=OCCURRENCES_FORMAT, project=None,
                 filters=None, interval=5, timeout=1800):
    """
    Request, wait for, and download an export in one call -- the whole
    round trip, which is what a pipeline script actually wants.

    Split into its four parts above as well, because a long export is worth
    being able to resume: if polling times out, the export is still running
    server-side and can be collected later with poll_export/download_export
    using its id.
    """
    export_id = request_export(session, fmt=fmt, project=project, filters=filters)
    export = poll_export(session, export_id, interval=interval, timeout=timeout)
    return download_export(session, export, dest)
