"""Antenna API transport: log in, request and download exports, page through list endpoints."""

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_BASE = "https://api.antenna.insectai.org/api/v2"

# Whether .env has been read into the environment yet. Read LAZILY -- on the
# first call that needs a credential or a url, not at import -- because
# importing a module should not reach out and mutate os.environ for the whole
# process. It did once, and the consequence was that anything importing this
# module inherited whatever .env happened to be beside the working directory,
# whether it wanted Antenna or not.
_environment_loaded = False


def _load_environment():
    """Read `.env` into the environment once per process, without overriding variables already set."""
    global _environment_loaded
    if not _environment_loaded:
        load_dotenv()
        _environment_loaded = True


def base():
    """Return the Antenna API root, from `ANTENNA_BASE` or the public default."""
    _load_environment()
    # `or`, not a get() default: .env.example ships ANTENNA_BASE= with nothing
    # after it, which sets the variable to the empty string. A blank line in a
    # config file means "I did not set this", not "the API lives at ''".
    return os.environ.get("ANTENNA_BASE") or DEFAULT_BASE


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
    """Return the Antenna project id, from `ANTENNA_PROJECT_ID`.

    Args:
        default: Value returned when the variable isn't set.

    Raises:
        RuntimeError: If the variable isn't set and `default` is None.
    """
    _load_environment()
    value = os.environ.get("ANTENNA_PROJECT_ID")
    if value is not None:
        return int(value)
    if default is not None:
        return int(default)
    raise RuntimeError(
        "no Antenna project id -- set ANTENNA_PROJECT_ID in the environment or pass project_id explicitly"
    )


def get_session():
    """Return an authenticated `requests.Session`, logging in with `ANTENNA_EMAIL` and `ANTENNA_PW`."""
    _load_environment()
    email = os.environ.get("ANTENNA_EMAIL")
    password = os.environ.get("ANTENNA_PW")
    if not (email and password):
        raise RuntimeError(
            "missing credentials -- set ANTENNA_EMAIL and ANTENNA_PW in the "
            "environment (a .env file for local work, or an EnvironmentFile on "
            "a server)"
        )

    session = requests.Session()
    response = session.post(
        f"{base()}/auth/token/login/", json={"email": email, "password": password}, timeout=30
    )
    response.raise_for_status()
    session.headers["Authorization"] = f"Token {response.json()['auth_token']}"
    logger.info("authenticated with Antenna as %s", email)
    return session


def paginate(session, path, params=None, page_size=PAGE_SIZE, timeout=60):
    """Yield every result across the pages of a list endpoint, following each response's `next`.

    Args:
        session: Authenticated session from `get_session()`.
        path: Endpoint path relative to the API root, e.g. `"captures/"`.
        params: Query parameters for the first request.
        page_size: Results asked for per page.
        timeout: Request timeout in seconds.
    """
    url = f"{base()}/{path}"
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
            logger.info(
                "%s: %d record(s) after %d pages, %.0fs elapsed", path, seen, pages, time.time() - started
            )

        seen += len(page["results"])
        yield from page["results"]

        url = page.get("next")
        params = None

    logger.info("%s: %d record(s) in %d page(s), %.0fs", path, seen, pages, time.time() - started)


# Query parameter that narrows captures to one event. The name matters more
# than it looks: the endpoint takes `event`, and an unrecognised filter is
# DROPPED SILENTLY rather than rejected -- `?event_id=12835` returns the whole
# project's 3,451 captures with a 200, not an error. So a plausible-looking
# capture from the wrong night is what a typo here buys you, which is why
# callers verify the event on what comes back (see calibrations/scale.py).
EVENT_PARAM = "event"


def fetch_captures(session, project=None, event=None, page_size=PAGE_SIZE):
    """Return the captures (light-trap sheet images) of a project, as raw API records.

    Each record carries its event and a presigned image URL.

    Args:
        session: Authenticated session.
        project: Antenna project id; from the environment if None.
        event: Narrow to one event's captures, which is one request instead of a walk of
            the whole project.
        page_size: Results asked for per page.
    """
    params = {"project_id": project_id(project)}
    if event is not None:
        params[EVENT_PARAM] = event

    return paginate(session, "captures/", params, page_size=page_size)


def request_export(session, fmt=OCCURRENCES_FORMAT, project=None, filters=None):
    """Request a new export, and return its id.

    Args:
        session: Authenticated session.
        fmt: Registered export format, e.g. `OCCURRENCES_FORMAT`.
        project: Antenna project id; from the environment if None.
        filters: Server-side filter dict. Only `collection_id` is supported.
    """
    project = project_id(project)
    payload = {"project": project, "format": fmt}
    if filters:
        payload["filters"] = filters

    response = session.post(f"{base()}/exports/", json=payload, timeout=30)
    response.raise_for_status()
    export_id = response.json()["id"]

    logger.info("requested %s export %s for project %s", fmt, export_id, project)
    return export_id


def poll_export(session, export_id, interval=5, timeout=1800):
    """Poll until an export is ready, and return its metadata dict, including `file_url`.

    Args:
        session: Authenticated session.
        export_id: Id from `request_export()`.
        interval: Seconds between polls.
        timeout: Seconds after which to raise. The export keeps running on the server and
            can be fetched later by id.
    """
    started = time.time()
    while time.time() - started < timeout:
        response = session.get(f"{base()}/exports/{export_id}/", timeout=30)
        response.raise_for_status()
        export = response.json()

        if export.get("file_url"):
            return export

        logger.info("export %s progress: %s", export_id, (export.get("job") or {}).get("progress"))
        time.sleep(interval)

    raise TimeoutError(f"export {export_id} not ready after {timeout}s")


def download_export(session, export, dest):
    """Stream a finished export to disk, and return the path.

    Args:
        session: Authenticated session.
        export: Export metadata dict from `poll_export()`.
        dest: Local path to write to.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with session.get(export["file_url"], stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)

    logger.info("downloaded export -> %s (%s records)", dest, export.get("record_count"))
    return dest


def fetch_export(session, dest, fmt=OCCURRENCES_FORMAT, project=None, filters=None, interval=5, timeout=1800):
    """Request, wait for and download an export in one call.

    Args:
        session: Authenticated session.
        dest: Local path to write to.
        fmt: As in `request_export`.
        project: As in `request_export`.
        filters: As in `request_export`.
        interval: As in `poll_export`.
        timeout: As in `poll_export`.
    """
    export_id = request_export(session, fmt=fmt, project=project, filters=filters)
    export = poll_export(session, export_id, interval=interval, timeout=timeout)
    return download_export(session, export, dest)
