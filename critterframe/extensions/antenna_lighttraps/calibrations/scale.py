"""
Physical scale for MothBox light traps, recovered from the reference card.

The Antenna way of producing the calibration core defines in
calibrations.scale: same calibration_type, same table, same parameters, a
source-specific way of measuring it. The matching is generic and stays in core
(calibrations.scale.scale_from_target). What is Antenna's, and therefore what is
here, is knowing that the card is a 1-inch circle in the top-left quadrant of a
sheet image, and that one card serves a whole trap night.

That division is why this module sits in the extension's calibrations/ rather
than its records/: it defines no table and owns no schema. It used to, back when
scale lived in an extension-local scalebars.parquet, and records/ was the right
home then. Now that the table is core's, all that's left here is a measurement
procedure.

Sheet images come from Antenna's captures endpoint, not from S3 directly. A
capture record carries a PRESIGNED url -- Antenna signs it, so there are no AWS
credentials, no bucket, and no session-path parsing here anymore. The captures
are also already tagged with their event, so grouping a night's sheets is a
matter of reading capture.event.id rather than deriving a date folder from a
detection URL.

ONE SCALE PER EVENT -- one trap night. The card is set up when a trap is
deployed and doesn't move until it's collected, so measuring it per detection
would be thousands of redundant measurements of one number, and a noisier answer
than measuring it once well. Per-machine would be wrong for the opposite reason:
a box picked up and put back down is a new setup and a new calibration.

The scope is `event_id`, Antenna's own identifier for a night. It used to key on
a `session_path` derived from the date folder in a detection's URL, which meant a
night running past midnight became two keys for one unmoved card (10 of 19 nights
in the project this was found on), and a URL the parser couldn't read became an
occurrence that could never be calibrated. event_id has neither problem: it comes
straight off the capture (and the occurrence) with no parsing, no midnight split,
nothing unparseable.

All of that is expressed as data, not code. Deciding later that the camera
drifts within a night means writing rows keyed on something narrower; nothing
here would need to change.

Nothing is written onto the occurrence table, which is the other reason the old
scalebars.parquet arrangement had to go: that table is a full snapshot, so every
re-ingest silently erased the calibration merged into it -- on a source whose
whole workflow is scheduled re-ingest. records.calibrations is keyed
independently of occurrences and survives them being replaced.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from ....calibrations import scale as scale_calibration
from ....records import calibrations as calibration_records
from ....visualization.panels import save_panel
from .. import api

logger = logging.getLogger(__name__)

# Real diameter of the quadrant target on the reference card.
CIRCLE_DIAMETER_MM = 25.4   # 1 inch

# The card always sits in the top-left quadrant of a sheet, so only that is
# searched -- fractional, so it holds if the camera resolution changes.
CARD_REGION = (0.0, 0.0, 0.5, 0.5)

# event_id is the calibration's scope: Antenna's own identifier for one trap
# night. No parsing, no midnight split, nothing unparseable -- it comes straight
# off the capture record and the occurrence table.
SCOPE_COL = "event_id"
SOURCE = "antenna_card"


def template_path(project_path):
    """
    The reference-card template image for a project.

    MUST be cropped tightly to the target circle's outer edge -- diameter is
    derived from the matched template width, so any padding around the circle
    becomes scale error directly.
    """
    return Path(project_path) / "mm2_scale_target_template.png"


def load_template(project_path):
    """Read a project's target template as grayscale, raising if it's missing."""
    template = cv2.imread(str(template_path(project_path)), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(
            f"no scale target template at {template_path(project_path)} -- put "
            "a tightly-cropped image of the card's quadrant target there"
        )
    return template


def pending_events(project_path, limit=None):
    """Events in the occurrence table with no scale measured yet."""
    return scale_calibration.pending_scope_values(project_path, SCOPE_COL,
                                                  limit=limit)


def first_sheet_for_event(session, event, project=None):
    """
    (capture_id, presigned_url) for one of an event's sheet images, or None.

    One request per event rather than a walk of the whole project: a night's
    captures are asked for directly, and the first one carrying a url is
    returned without paging any further. The alternative -- fetching all 3,451
    captures to use 19 of them -- is what made a scale pass take minutes with
    nothing to show for it.

    VERIFIES THE EVENT ON WHAT COMES BACK, which is not paranoia. The captures
    endpoint drops an unrecognised filter silently: ask it for `event_id=12835`
    rather than `event=12835` and it returns the entire project with a 200. The
    first capture of that would be some arbitrary other night, and calibrating
    one night from another night's sheet is a wrong answer that looks entirely
    reasonable in the table afterwards. So it's checked, and it raises.
    """
    for capture in api.fetch_captures(session, project=project, event=event):
        returned = (capture.get("event") or {}).get("id")
        if str(returned) != str(event):
            raise ValueError(
                f"asked the captures endpoint for event {event} and got a "
                f"capture from event {returned} -- the event filter is being "
                f"ignored, so every scale measured this way would be from an "
                f"arbitrary night. Check api.EVENT_PARAM against the API."
            )
        if capture.get("url"):
            return capture.get("id"), capture["url"]

    return None


def _download_sheet(url, session=None, timeout=300):
    """
    Fetch one sheet image's bytes from its presigned url and decode to BGR.

    A plain unauthenticated GET -- the url carries its own auth in the query
    string, which is the whole point of routing through the captures endpoint.

    session -- optional requests session to fetch through. The url needs no
               authentication, so this is not about credentials: it is so one
               session covers a whole scale pass (connection reuse over twenty
               20 MB downloads is not nothing), and so a caller can hand in
               something else entirely. Every other network call in the package
               takes one; this is the last that did not.

    These are full-resolution sheets, around 20 MB each, and that download is
    the slow part of a scale pass now that the metadata isn't. Logged with its
    size and duration, because a minute of silence per event is what sent
    someone looking for a hang last time. The timeout is generous for the same
    reason: a slow 20 MB is normal here, not a fault.
    """
    started = time.perf_counter()
    response = (session or requests).get(url, timeout=timeout)
    response.raise_for_status()

    image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the sheet image")

    height, width = image.shape[:2]
    logger.info("  fetched %.1f MB in %.1fs -> %dx%d",
                len(response.content) / 1e6, time.perf_counter() - started,
                width, height)
    return image


def measure_scales(project_path, project=None, limit=None, visualize=False,
                   session=None):
    """
    Measure scale for every event that doesn't have one yet, and record it.

    Named to mirror calibrations.scale.measure_scales, which does the same job
    for a target in the occurrence's own image: same calibration produced, same
    repeat-awareness, different place to find the card.

    For each pending event, asks the captures endpoint for that event's sheets,
    downloads ONE of them from its presigned url straight into memory -- no
    temporary files -- and matches the target on it. Individual failures are
    logged and don't stop the remaining events, and already-measured events are
    skipped, so this is safe to rerun as new deployments arrive.

    Progress is logged per event, with the size and duration of each download.
    That isn't decoration: a sheet is around 20 MB, so a pass over twenty events
    is minutes of mostly-waiting, and the version of this that said nothing
    while it worked was indistinguishable from a hang.

    One sheet per event, deliberately: see the module docstring. If an event's
    calibration is ever in doubt, the diagnostic is to measure several of its
    sheets by hand with calibrations.scale.scale_from_target and compare -- a
    spread means the camera moved during the night.

    project -- Antenna project id to read captures from; taken from the
               environment (ANTENNA_PROJECT_ID) if omitted, the same way every
               other Antenna call resolves it.
    session -- authenticated session to use; one is created if omitted. Passing
               one lets a script share it with a download pass, and is what
               makes this function reachable without credentials.

    Returns a summary dict (saved, failed, unaddressable).
    """
    template = load_template(project_path)

    pending = pending_events(project_path, limit=limit)
    if not pending:
        logger.info("every event already has a scale -- nothing to measure")
        return {"saved": 0, "failed": 0, "unaddressable": 0}

    session = session or api.get_session()
    logger.info("%d event(s) pending scale measurement; one sheet image each, "
                "around 20 MB apiece", len(pending))

    rows = []
    failed = 0
    unaddressable = 0
    started = time.perf_counter()

    for index, event in enumerate(pending, start=1):
        logger.info("event %s (%d of %d)", event, index, len(pending))

        try:
            sheet = first_sheet_for_event(session, event, project=project)
        except Exception as exc:
            failed += 1
            logger.warning("  could not look up sheets for event %s: %s", event, exc)
            continue

        # No captures for this event means Antenna has no sheet image for the
        # night -- nothing to measure. Reported, not counted as a failure,
        # because there is nothing here to fix.
        if sheet is None:
            unaddressable += 1
            logger.info("  no sheet image for this event; skipping")
            continue

        capture_id, url = sheet

        try:
            image = _download_sheet(url, session=session)

            result = scale_calibration.scale_from_target(
                image, template, CIRCLE_DIAMETER_MM, region=CARD_REGION,
                name=str(capture_id))
            if result is None:
                raise ValueError("no scale target detected")

            if visualize:
                save_panel(project_path,
                           scale_calibration.scale_panel(image, result),
                           str(capture_id), subdir="scale")

            rows.append(scale_calibration.make_scale_row(
                SCOPE_COL, event, result["px_per_mm"], source=SOURCE,
                score=result["score"], measured_from=str(capture_id)))

        except Exception as exc:
            failed += 1
            logger.warning("  scale measurement failed for event %s: %s", event, exc)

    calibration_records.save_calibrations(project_path, rows)
    logger.info("event scale pass complete in %.0fs: saved=%d failed=%d "
                "unaddressable=%d", time.perf_counter() - started, len(rows),
                failed, unaddressable)
    return {"saved": len(rows), "failed": failed,
            "unaddressable": unaddressable}