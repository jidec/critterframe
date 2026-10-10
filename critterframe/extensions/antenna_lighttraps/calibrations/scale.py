"""Scale for Antenna light traps: px/mm from the reference card on a sheet image, one per event."""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ....ingest import download as core_download
from ....calibrations import scale as scale_calibration
from ....core.recipes import hash_spec
from ....records import calibrations as calibration_records
from ....visualization import figures
from ....visualization import pipeline as pipeline_visualization
from .. import api

logger = logging.getLogger(__name__)

# Real diameter of the quadrant target on the reference card.
CIRCLE_DIAMETER_MM = 25.4  # 1 inch

# The card always sits in the top-left quadrant of a sheet, so only that is
# searched -- fractional, so it holds if the camera resolution changes.
CARD_REGION = (0.0, 0.0, 0.5, 0.5)

# event_id is the calibration's scope: Antenna's own identifier for one trap
# night. No parsing, no midnight split, nothing unparseable -- it comes straight
# off the capture record and the occurrence table.
SCOPE_COL = "event_id"
SOURCE = "antenna_card"

# (connect, read) timeout for one sheet. Long, because a sheet is around 20 MB
# and a slow one here is normal rather than a fault.
SHEET_TIMEOUT = (10, 300)


def template_path(project_path):
    """Return the path of a project's reference-card template image.

    The template must be cropped tight to the card's outer edge: any padding becomes
    scale error.
    """
    return Path(project_path) / "mm2_scale_target_template.png"


def load_template(project_path):
    """Read a project's card template as grayscale, raising if it is missing."""
    template = cv2.imread(str(template_path(project_path)), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(
            f"no scale target template at {template_path(project_path)} -- put "
            "a tightly-cropped image of the card's quadrant target there"
        )
    return template


def _pending_events(project_path, max_new=None):
    """Return the events in the occurrence table with no scale yet."""
    return scale_calibration.pending_scope_values(project_path, SCOPE_COL, max_new=max_new)


def _first_sheet_for_event(session, event, project=None):
    """Return `(capture_id, presigned_url)` for one of an event's sheet images, or None.

    Raises if a returned capture belongs to another event: the endpoint ignores a filter
    it doesn't recognize and returns the whole project.
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


def _download_sheet(url, session=None, timeout=SHEET_TIMEOUT):
    """Fetch one sheet image from its presigned URL and decode it to BGR.

    Args:
        url: Presigned URL from the captures endpoint.
        session: Session to fetch through. Use an unauthenticated one: the URL carries
            its own credentials, and storage can reject an added Authorization header.
        timeout: `(connect, read)` timeout.
    """
    started = time.perf_counter()
    content = core_download._download_image(url, session or core_download.make_session(), timeout=timeout)

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the sheet image")

    height, width = image.shape[:2]
    logger.info(
        "  fetched %.1f MB in %.1fs -> %dx%d",
        len(content) / 1e6,
        time.perf_counter() - started,
        width,
        height,
    )
    return image


def measure_scales(project_path, project=None, max_new=None, visualize=True, session=None):
    """Measure a scale for every event that has none, and record it.

    One sheet per event is downloaded into memory and the card matched on it. A failed
    event is logged and the rest continue.

    Args:
        project_path: Project to measure and record in.
        project: Antenna project id; from the environment if None.
        max_new: Cap on how many events are measured in this call.
        visualize: True, an int, or event ids: a pipeline grid of the weakest card
            matches and a px/mm histogram. False writes nothing.
        session: Authenticated session to reuse.

    Returns:
        A summary dict with `saved`, `failed` and `unaddressable`.
    """
    template = load_template(project_path)

    pending = _pending_events(project_path, max_new=max_new)
    if not pending:
        logger.info("every event already has a scale -- nothing to measure")
        return {"saved": 0, "failed": 0, "unaddressable": 0}

    session = session or api.get_session()
    # Two sessions on purpose: the API one is authenticated, while a presigned
    # storage url carries its own credentials in the query string and can be
    # rejected for presenting a second set.
    storage_session = core_download.make_session()
    logger.info(
        "%d event(s) pending scale measurement; one sheet image each, around 20 MB apiece", len(pending)
    )

    identity = {
        "kind": "antenna_measure_scales",
        "template": scale_calibration.image_digest(template),
        "target_mm": CIRCLE_DIAMETER_MM,
        "region": list(CARD_REGION),
        "scope": SCOPE_COL,
        "source": SOURCE,
    }
    report = pipeline_visualization.open_report(
        project_path,
        "measure_scales__antenna",
        hash_spec(identity),
        visualize=visualize,
        rank="lowest",
        identity=identity,
    ).begin([str(event) for event in pending])

    rows = []
    measured_px = []
    failed = 0
    unaddressable = 0
    started = time.perf_counter()

    for index, event in enumerate(pending, start=1):
        logger.info("event %s (%d of %d)", event, index, len(pending))
        item = str(event)

        try:
            sheet = _first_sheet_for_event(session, event, project=project)
        except Exception as exc:
            failed += 1
            report.failure(item, exc)
            report.done(item)
            logger.warning("  could not look up sheets for event %s: %s", event, exc)
            continue

        # No captures for this event means Antenna has no sheet image for the
        # night -- nothing to measure. Reported, not counted as a failure,
        # because there is nothing here to fix.
        if sheet is None:
            unaddressable += 1
            report.done(item)
            logger.info("  no sheet image for this event; skipping")
            continue

        capture_id, url = sheet
        score = None

        try:
            image = _download_sheet(url, session=storage_session)

            result = scale_calibration.scale_from_target(
                image, template, CIRCLE_DIAMETER_MM, region=CARD_REGION, name=str(capture_id)
            )
            if result is None:
                raise ValueError("no scale target detected")

            if report.wants(item):
                report.panel(item, "card", _card_panel(image, result))

            rows.append(
                scale_calibration.make_scale_row(
                    SCOPE_COL,
                    event,
                    result["px_per_mm"],
                    source=SOURCE,
                    score=result["score"],
                    measured_from=str(capture_id),
                )
            )
            measured_px.append(result["px_per_mm"])
            score = result["score"]

        except Exception as exc:
            failed += 1
            report.failure(item, exc)
            logger.warning("  scale measurement failed for event %s: %s", event, exc)

        report.done(item, rank_value=score)

    if report and measured_px:
        report.figure(
            "px_per_mm",
            figures.histogram(
                measured_px, bins=20, xlabel="px/mm", title=f"card scale per event (n={len(measured_px)})"
            ),
        )
    report.close()

    calibration_records.save_calibrations(project_path, rows)
    logger.info(
        "event scale pass complete in %.0fs: saved=%d failed=%d unaddressable=%d",
        time.perf_counter() - started,
        len(rows),
        failed,
        unaddressable,
    )
    return {"saved": len(rows), "failed": failed, "unaddressable": unaddressable}


def _card_panel(image, result):
    """Return the matched card drawn on its sheet, cropped to the quadrant searched."""
    panel = scale_calibration.scale_panel(image, result)
    height, width = panel.shape[:2]
    left, top, right, bottom = CARD_REGION
    return panel[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
