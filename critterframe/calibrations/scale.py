"""Scale calibration: px/mm from a target of known size, matched automatically or clicked by hand."""

import hashlib
import logging
import time

import cv2
import numpy as np
import pandas as pd

from .. import drivers
from ..project import paths, subsets as subset_selection
from ..recipes import hash_spec
from ..records import calibrations as calibration_records
from ..records.occurrences import ID_COL
from ..storage.imagestore import ImageStore
from ..visualization import figures
from ..visualization import pipeline as pipeline_visualization
from ..visualization.panels import annotate, fit_for_display

logger = logging.getLogger(__name__)

# This module's calibration_type in the shared table, and the one parameter a
# scale currently carries. Named constants because the extension writes rows too
# and a typo in either would be a silently empty resolution rather than an error.
CALIBRATION_TYPE = "scale"
SCALE_COL = "px_per_mm"

# What scale_from_click() fits its window to by default: the shared screen box
# (panels.DISPLAY_MAX). A full-resolution scene image (a light-trap sheet,
# easily 4000px+) routinely exceeds any screen, and cv2 does not scale a window
# on its own, so without this the scale bar can end up off-screen.
DEFAULT_MAX_DISPLAY = "screen"

# Scales to try the template at, as multiples of its own size. Wide because a
# template cropped from one project's photo may meet a camera at a different
# resolution entirely.
DEFAULT_SCALES = np.linspace(0.5, 2.0, 31)

# Minimum normalized-correlation peak (TM_CCOEFF_NORMED, -1..1) to accept a
# match. Real target peaks sit around 0.6+; raise it if false matches pass,
# lower it if a valid target is rejected under poor lighting.
MATCH_SCORE_MIN = 0.4

# Peak below which an accepted match is worth a second look. Between this and
# MATCH_SCORE_MIN is the band where clutter can out-correlate an absent target
# -- textured background in roughly the right size range will do it -- and the
# result is the dangerous kind of wrong: a plausible number, silently applied to
# every trait it calibrates. Accepted anyway, because a genuine target in bad
# light also lands here and refusing it would lose real data, but said out loud.
WEAK_MATCH_SCORE = 0.6


def _match_at_scales(gray, template, scales):
    """Match a template at several scales.

    Returns:
        `(cx, cy, radius, score, matched_width)` in `gray`'s frame for the scale with the
        highest correlation peak, or None if nothing fits.
    """
    template_height, template_width = template.shape[:2]  # get height and width
    best = None

    for scale in scales:  # for each scale
        width, height = int(template_width * scale), int(template_height * scale)  # get scaled h/w
        if width < 8 or height < 8 or height > gray.shape[0] or width > gray.shape[1]:
            continue

        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)  # resize
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)  # match template
        _, peak, _, location = cv2.minMaxLoc(result)

        if best is None or peak > best[3]:
            # the template is cropped to the target, so half its width is the radius
            best = (location[0] + width // 2, location[1] + height // 2, width // 2, peak, width)

    return best


def _region_slice(image, region):
    """Return the sub-image a fractional `(x0, y0, x1, y1)` region selects, and its offset in the frame."""
    if region is None:
        return image, (0, 0)

    height, width = image.shape[:2]
    x0, y0, x1, y1 = region
    left, top = int(x0 * width), int(y0 * height)
    right, bottom = int(x1 * width), int(y1 * height)

    if right - left < 8 or bottom - top < 8:
        raise ValueError(f"region {region} is smaller than the smallest template the matcher will try")
    return image[top:bottom, left:right], (left, top)


def _detect_target(
    image, template, region=None, coarse_scales=DEFAULT_SCALES, match_score_min=MATCH_SCORE_MIN
):
    """Find a target by multi-scale template matching.

    Returns:
        `(cx, cy, radius, score)` in full-image coordinates, or None.
    """
    searched, (offset_x, offset_y) = _region_slice(image, region)
    gray = searched if searched.ndim == 2 else cv2.cvtColor(searched, cv2.COLOR_BGR2GRAY)

    coarse = _match_at_scales(gray, template, coarse_scales)
    if coarse is None or coarse[3] < match_score_min:
        return None

    won_scale = coarse[4] / template.shape[1]
    step = coarse_scales[1] - coarse_scales[0]
    fine = _match_at_scales(gray, template, np.linspace(won_scale - step, won_scale + step, 21))

    best = fine if (fine is not None and fine[3] >= coarse[3]) else coarse
    cx, cy, radius, score, _ = best
    return cx + offset_x, cy + offset_y, radius, score


def scale_from_target(
    image,
    template,
    target_mm,
    region=None,
    coarse_scales=DEFAULT_SCALES,
    match_score_min=MATCH_SCORE_MIN,
    name=None,
):
    """Measure pixels per millimeter from an image containing a target of known width.

    A high match score is not evidence the scale is right: clutter can out-correlate an
    absent target, so check the panel.

    Args:
        image: BGR or grayscale array showing the target.
        template: Grayscale template, cropped tight to the target's outer edge. The matched
            width is the measurement, so any margin is measured too.
        target_mm: The target's real width in millimeters.
        region: Fractional `(x0, y0, x1, y1)` box to search in; None searches the whole frame.
        coarse_scales: Template scales to try.
        match_score_min: Match score below which the result is flagged as weak.
        name: Label for the log line.

    Returns:
        The measurement as a dict, or None if no target was found.
    """
    started = time.perf_counter()
    found = _detect_target(
        image, template, region=region, coarse_scales=coarse_scales, match_score_min=match_score_min
    )
    elapsed = time.perf_counter() - started
    if found is None:
        logger.warning(
            "no scale target detected%s after %.1fs -- if it's "
            "visible in the frame, try lowering match_score_min, "
            "widening coarse_scales, or checking region",
            f" in {name}" if name else "",
            elapsed,
        )
        return None

    cx, cy, radius, score = found
    diameter_px = 2 * radius
    px_per_mm = diameter_px / float(target_mm)

    # The elapsed time is worth reporting: a full-resolution light-trap sheet
    # takes tens of seconds to sweep, which is the dominant cost of a scale pass
    # and not at all obvious from the outside.
    logger.info(
        "%s: target at (%d,%d) r=%dpx match=%.3f -> %.4f px/mm (%.1fs)",
        name or "image",
        cx,
        cy,
        radius,
        score,
        px_per_mm,
        elapsed,
    )
    if score < WEAK_MATCH_SCORE:
        logger.warning(
            "%s: weak match (%.3f) -- this may be clutter rather "
            "than the target, and a wrong scale is worse than none. "
            "Check the panel, or raise match_score_min",
            name or "image",
            score,
        )

    return {
        "px_per_mm": float(px_per_mm),
        "score": float(score),
        "cx": int(cx),
        "cy": int(cy),
        "radius_px": int(radius),
        "diameter_px": int(diameter_px),
    }


def scale_panel(image, result):
    """Return the image with the matched circle, its center and the numbers drawn on it."""
    panel = np.asarray(image).copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    cv2.circle(panel, (result["cx"], result["cy"]), result["radius_px"], (0, 255, 0), 2)
    cv2.circle(panel, (result["cx"], result["cy"]), 2, (0, 0, 255), 3)
    annotate(panel, f"{result['px_per_mm']:.4f} px/mm  match {result['score']:.3f}")
    return panel


def scale_from_click(image, target_mm=None, name=None, max_display=DEFAULT_MAX_DISPLAY):
    """Measure pixels per millimeter from two points a person clicks, a known length apart.

    Left-click the two ends in either order. Esc, or a blank or non-positive typed length,
    cancels.

    Args:
        image: BGR or grayscale array showing the scale object, usually loaded by the caller.
        target_mm: The object's real length in millimeters. None prompts for it on the
            terminal once both points are clicked.
        name: Shown in the window title and the log line.
        max_display: `"screen"` fits the window to `panels.DISPLAY_MAX`, an int is the longest
            side it is shrunk to, None shows full resolution. Clicks are mapped back to the
            image's own pixels in every case.

    Returns:
        `{px_per_mm, length_px, point_a, point_b}`, or None if cancelled.
    """
    original = np.asarray(image)
    height, width = original.shape[:2]
    display = original
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    if isinstance(max_display, (int, float)):
        display, display_scale = fit_for_display(display, (max_display, max_display), enlarge=False)
    else:
        display, display_scale = fit_for_display(display, max_display)
    display = display.copy()

    window = f"{name or 'scale'} - click the two ends of the scale object (Esc=cancel)"
    cv2.imshow(window, display)

    display_points = []  # in the (possibly shrunk) DISPLAYED frame

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(display_points) < 2:
            display_points.append((x, y))
            color = (0, 255, 0) if len(display_points) == 1 else (0, 0, 255)
            cv2.circle(display, (x, y), 6, color, -1)
            if len(display_points) == 2:
                cv2.line(display, display_points[0], display_points[1], (0, 255, 255), 2)
            cv2.imshow(window, display)

    cv2.setMouseCallback(window, on_click)

    cancelled = False
    while len(display_points) < 2:
        if cv2.waitKey(20) & 0xFF == 27:
            cancelled = True
            break
    cv2.destroyWindow(window)

    if cancelled:
        logger.info("%s: scale click cancelled", name or "image")
        return None

    # Back to ORIGINAL-image pixels -- the only frame px_per_mm can mean
    # anything in, since that's the frame every other measurement is made in.
    (x0, y0), (x1, y1) = (
        (min(width - 1, max(0, round(x / display_scale))), min(height - 1, max(0, round(y / display_scale))))
        for x, y in display_points
    )
    length_px = float(np.hypot(x1 - x0, y1 - y0))

    if target_mm is None:
        target_mm = _prompt_length_mm(name)
    if target_mm is None:
        return None

    px_per_mm = length_px / float(target_mm)
    logger.info(
        "%s: clicked %.1fpx over %gmm -> %.4f px/mm", name or "image", length_px, target_mm, px_per_mm
    )

    return {"px_per_mm": float(px_per_mm), "length_px": length_px, "point_a": [x0, y0], "point_b": [x1, y1]}


def _prompt_length_mm(name=None):
    """Ask on the terminal for the scale object's length; None for a blank or non-positive answer."""
    raw = input(f"{name or 'image'}: length of the clicked scale object, in mm: ")
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("%s: %r is not a number -- scale not recorded", name or "image", raw)
        return None
    if not value > 0:
        logger.warning("%s: length must be positive, got %s -- scale not recorded", name or "image", raw)
        return None
    return value


def click_scale_panel(image, result):
    """Return the image with the two clicked points, the line between them and the px/mm drawn on it."""
    panel = np.asarray(image).copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    point_a, point_b = tuple(result["point_a"]), tuple(result["point_b"])
    cv2.line(panel, point_a, point_b, (0, 255, 255), 2)
    cv2.circle(panel, point_a, 6, (0, 255, 0), -1)
    cv2.circle(panel, point_b, 6, (0, 0, 255), -1)
    annotate(panel, f"{result['px_per_mm']:.4f} px/mm  ({result['length_px']:.1f}px)")
    return panel


def make_scale_row(scope, scope_value, px_per_mm, source, score=None, measured_from=None):
    """Build one scale calibration record, ready for `records.calibrations.save_calibrations`.

    Args:
        scope: Occurrence column the record applies across.
        scope_value: The value in that column.
        px_per_mm: Pixels per millimeter.
        source: How it was obtained: `"target"`, `"clicked"` or `"declared"`.
        score: Template match score, where there was one.
        measured_from: What it was measured on, e.g. an image digest or a name.

    Raises:
        ValueError: If `px_per_mm` is not positive.
    """
    px_per_mm = float(px_per_mm)
    if not px_per_mm > 0:
        raise ValueError(f"px_per_mm must be positive, got {px_per_mm} for {scope}={scope_value!r}")

    return calibration_records.make_calibration_row(
        CALIBRATION_TYPE,
        scope,
        scope_value,
        {SCALE_COL: px_per_mm},
        source=source,
        score=score,
        measured_from=measured_from,
    )


def declare_scale(project_path, px_per_mm, scope=ID_COL, scope_value=None, measured_from=None):
    """Record a scale that is already known, e.g. a scanner's dpi or a fixed copy stand.

    Args:
        project_path: Project to record it in.
        px_per_mm: Pixels per millimeter; a scanner's is `dpi / 25.4`.
        scope: Occurrence column it applies across; `ID_COL` means one occurrence.
        scope_value: The value in that column. Required.
        measured_from: Where the figure comes from.
    """
    if scope_value is None:
        raise ValueError(
            "declare_scale needs a scope_value -- which occurrence, session, or "
            f"device is {px_per_mm} px/mm true of?"
        )

    row = make_scale_row(scope, scope_value, px_per_mm, source="declared", measured_from=measured_from)
    calibration_records.save_calibrations(project_path, [row])
    logger.info("declared %.4f px/mm for %s=%s", px_per_mm, scope, scope_value)
    return row


def scale_for_occurrences(project_path, occurrence_ids=None):
    """Return px/mm per occurrence, as a float Series indexed by occurrence id.

    Args:
        project_path: Project to read from.
        occurrence_ids: Occurrences to resolve; all if None.

    Returns:
        The narrowest-scoped scale covering each occurrence, NaN where none does.
    """
    resolved = calibration_records.resolve_for_occurrences(
        project_path, CALIBRATION_TYPE, occurrence_ids=occurrence_ids
    )
    if resolved.empty:
        return pd.Series(dtype="float64", name=SCALE_COL)

    values = resolved.map(
        lambda parameters: parameters.get(SCALE_COL) if isinstance(parameters, dict) else None
    )
    return values.astype("float64").rename(SCALE_COL)


def pending_scope_values(project_path, scope, max_new=None):
    """Return the scope values with no scale calibration yet, at most `max_new` of them."""
    return calibration_records.pending_scope_values(project_path, CALIBRATION_TYPE, scope, max_new=max_new)


def _measured_values(project_path, scope):
    """Return the scope values that already have a scale, as a set of strings."""
    measured = calibration_records.load_calibrations(
        project_path, calibration_type=CALIBRATION_TYPE, scope=scope
    )
    return set(measured["scope_value"].astype(str))


def image_digest(image):
    """Return a short digest of an array's shape and pixels.

    Args:
        image: Any NumPy array.
    """
    array = np.ascontiguousarray(image)
    digest = hashlib.sha256(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def measure_scales(
    project_path,
    template,
    target_mm,
    scope=ID_COL,
    region=None,
    match_score_min=MATCH_SCORE_MIN,
    subset=None,
    limit=None,
    max_new=None,
    force=False,
    visualize=True,
):
    """Measure a scale target in each occurrence's own image and record the results.

    Args:
        project_path: Project to measure and record in.
        template: Grayscale template, cropped tight to the target.
        target_mm: The target's real width in millimeters.
        scope: Occurrence column the records are keyed on. A grouping column records one
            measurement as covering the whole group.
        region: Fractional `(x0, y0, x1, y1)` box to search in.
        match_score_min: Match score below which a result is flagged as weak.
        subset: Named subset to restrict to.
        limit: Cap on the occurrences considered, before already-measured ones are excluded.
        max_new: Cap on how many are measured in this call.
        force: Re-measure scope values that already have a scale.
        visualize: True, an int, or scope values: a pipeline grid of the weakest matches and
            a px/mm histogram. False writes nothing.

    Returns:
        The `drivers.Tally.summary` dict plus `missed`, the images where no target matched.
    """
    paths.require_project(project_path)

    calibration_records.require_scope_column(project_path, scope)
    occurrences = subset_selection.select_occurrences(
        project_path, subset=subset, limit=limit, columns=[scope]
    )

    done = set() if force else _measured_values(project_path, scope)

    # One measurement per scope value: with the default ID_COL scope that's one
    # per occurrence, and with a grouping scope it's the first occurrence of
    # each group, since the rest would re-measure the same physical setup.
    todo = []
    seen = set()
    for row in occurrences.itertuples(index=False):
        value = str(getattr(row, scope))
        if value in seen or value in done:
            continue
        seen.add(value)
        todo.append((getattr(row, ID_COL), value))
    if max_new is not None:
        todo = todo[:max_new]

    skipped = len(occurrences) - len(todo)
    logger.info(
        "measuring scale on %d image(s) keyed by '%s'; %d already covered or duplicated",
        len(todo),
        scope,
        skipped,
    )

    identity = {
        "kind": "measure_scales",
        "template": image_digest(template),
        "target_mm": target_mm,
        "scope": scope,
        "region": None if region is None else list(region),
        "match_score_min": match_score_min,
    }
    report = pipeline_visualization.open_report(
        project_path,
        "measure_scales",
        hash_spec(identity),
        visualize=visualize,
        rank="lowest",
        identity=identity,
    ).begin([value for _occurrence_id, value in todo])

    rows = []
    measured_px = []
    missed = 0
    tally = drivers.Tally(attempted=len(todo))
    tally.skipped = skipped

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id, value in todo:
            score = None
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                result = scale_from_target(
                    image,
                    template,
                    target_mm,
                    region=region,
                    match_score_min=match_score_min,
                    name=str(occurrence_id),
                )
                if result is None:
                    missed += 1
                    tally.no_input += 1
                    report.failure(value, f"no target matched in {occurrence_id}")
                else:
                    score = result["score"]
                    if report.wants(value):
                        report.panel(value, "scale", scale_panel(image, result))
                    rows.append(
                        make_scale_row(
                            scope,
                            value,
                            result["px_per_mm"],
                            source="target",
                            score=result["score"],
                            measured_from=str(occurrence_id),
                        )
                    )
                    measured_px.append(result["px_per_mm"])
                    tally.processed += 1

            except Exception as exc:
                tally.record_failure(value, exc)
                report.failure(value, exc)
                logger.warning("scale measurement failed for %s: %s", occurrence_id, exc)

            report.done(value, rank_value=score)

    if report and measured_px:
        report.figure(
            "px_per_mm",
            figures.histogram(
                measured_px, bins=20, xlabel="px/mm", title=f"scale by '{scope}' (n={len(measured_px)})"
            ),
        )
    report.close()

    calibration_records.save_calibrations(project_path, rows)
    logger.info(
        "scale pass complete: measured=%d skipped=%d failed=%d missed=%d",
        tally.processed,
        tally.skipped,
        tally.failed,
        missed,
    )
    return tally.summary(missed=missed)


def measure_scale_by_hand(
    project_path,
    image,
    target_mm=None,
    scope=ID_COL,
    scope_value=None,
    occurrence_ids=None,
    subset=None,
    source="clicked",
    name=None,
    visualize=True,
    force=False,
    max_display=DEFAULT_MAX_DISPLAY,
):
    """Click a known length in a standalone scene image and record the scale for many occurrences.

    Only occurrences or scope values with no scale yet are written, unless `force`.

    Args:
        project_path: Project to record the calibration in.
        image: BGR or grayscale array showing the scale object, loaded by the caller.
        target_mm: The object's real length in millimeters; None prompts for it.
        scope: Occurrence column the records are keyed on. `ID_COL` writes one per occurrence
            named by `occurrence_ids` or `subset`; another column writes one for `scope_value`.
        scope_value: The value in `scope` this measurement covers. Required unless `scope`
            is `ID_COL`, and rejected when it is.
        occurrence_ids: With `scope=ID_COL`, the occurrences covered.
        subset: With `scope=ID_COL`, a named subset to cover instead. With neither, every
            occurrence in the project.
        source: Recorded as the calibration's provenance.
        name: Shown in the window title and log line, and recorded as `measured_from`.
        visualize: Write a one-panel pipeline grid of the clicked points and the px/mm.
        force: Measure again and overwrite existing scales.
        max_display: As in `scale_from_click`.

    Returns:
        `scale_from_click`'s result with `covered` (records written) added, or None if
        cancelled or if everything named already had a scale.
    """
    paths.require_project(project_path)

    if scope != ID_COL:
        if occurrence_ids is not None or subset is not None:
            raise ValueError(
                "occurrence_ids/subset only apply with scope=ID_COL -- pass scope_value for any other scope"
            )
        if scope_value is None:
            raise ValueError(
                f"scope={scope!r} needs a scope_value -- which {scope} is this measurement true of?"
            )
        calibration_records.require_scope_column(project_path, scope)
        targets = [str(scope_value)]
    else:
        if scope_value is not None:
            raise ValueError(
                "scope_value doesn't apply with scope=ID_COL -- pass occurrence_ids or subset instead"
            )
        if occurrence_ids is not None and subset is not None:
            raise ValueError("pass occurrence_ids or subset, not both")
        if occurrence_ids is not None:
            targets = [str(occurrence_id) for occurrence_id in occurrence_ids]
        else:
            targets = subset_selection.select_ids(project_path, subset=subset)

    if force:
        skipped = 0
    else:
        measured = _measured_values(project_path, scope)
        pending = [value for value in targets if value not in measured]
        skipped = len(targets) - len(pending)
        targets = pending

    if not targets:
        logger.info("every requested '%s' value already has a scale -- nothing to measure", scope)
        return None

    result = scale_from_click(image, target_mm=target_mm, name=name, max_display=max_display)
    if result is None:
        return None

    if visualize:
        item = name or "scene"
        identity = {
            "kind": "measure_scale_by_hand",
            "image": image_digest(image),
            "name": name,
            "scope": scope,
            "scope_value": scope_value,
            "source": source,
            "target_mm": result.get("target_mm", target_mm),
        }
        with pipeline_visualization.open_report(
            project_path, "measure_scale_by_hand", hash_spec(identity), visualize=[item], identity=identity
        ).begin([item]) as report:
            report.panel(item, "clicked", click_scale_panel(image, result))
            report.done(item)

    rows = [
        make_scale_row(scope, value, result["px_per_mm"], source=source, measured_from=name)
        for value in targets
    ]
    calibration_records.save_calibrations(project_path, rows)

    logger.info(
        "recorded %.4f px/mm for %d '%s' value(s) (%d already covered)",
        result["px_per_mm"],
        len(rows),
        scope,
        skipped,
    )
    return {**result, "covered": len(rows)}
