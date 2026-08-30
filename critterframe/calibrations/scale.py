"""
px/mm from a target of known size in the frame.

The detector is generic: target, size, and search region are all arguments. A
weak match is accepted but warned about, since clutter can out-correlate an
absent target and a plausible wrong scale is worse than none -- a high match
score is NOT evidence the millimetres are right, so check the panel.

The template must be cropped tight to the target's outer edge: the matched
width IS the measurement, so margin in the template measures the margin too.
"""

import logging
import time

import cv2
import numpy as np
import pandas as pd

from ..project import paths, subsets as subset_selection
from ..records import calibrations as calibration_records
from ..records.occurrences import ID_COL
from ..storage.imagestore import ImageStore
from ..visualization.panels import annotate, save_panel

logger = logging.getLogger(__name__)

# This module's calibration_type in the shared table, and the one parameter a
# scale currently carries. Named constants because the extension writes rows too
# and a typo in either would be a silently empty resolution rather than an error.
CALIBRATION_TYPE = "scale"
SCALE_COL = "px_per_mm"

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
    """
    Multi-scale template match. Returns (cx, cy, radius, score, matched_width)
    in `gray`'s frame for the scale with the highest correlation peak, or None
    if nothing fits.
    """
    template_height, template_width = template.shape[:2]
    best = None

    for scale in scales:
        width, height = int(template_width * scale), int(template_height * scale)
        if width < 8 or height < 8 or height > gray.shape[0] or width > gray.shape[1]:
            continue

        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, peak, _, location = cv2.minMaxLoc(result)

        if best is None or peak > best[3]:
            # the template is cropped to the target, so half its width is the radius
            best = (location[0] + width // 2, location[1] + height // 2,
                    width // 2, peak, width)

    return best


def _region_slice(image, region):
    """
    The sub-image to search, and its offset in the full frame.

    region is fractional -- (x0, y0, x1, y1) in 0..1 -- so one setting survives
    a change of camera resolution, which pixel coordinates would not.
    """
    if region is None:
        return image, (0, 0)

    height, width = image.shape[:2]
    x0, y0, x1, y1 = region
    left, top = int(x0 * width), int(y0 * height)
    right, bottom = int(x1 * width), int(y1 * height)

    if right - left < 8 or bottom - top < 8:
        raise ValueError(f"region {region} is smaller than the smallest template "
                         "the matcher will try")
    return image[top:bottom, left:right], (left, top)


def detect_target(image, template, region=None, coarse_scales=DEFAULT_SCALES,
                  match_score_min=MATCH_SCORE_MIN):
    """
    Find a target by multi-scale template matching. Returns
    (cx, cy, radius, score) in FULL-image coordinates, or None.

    region -- fractional (x0, y0, x1, y1) box to search in, or None for the
              whole frame. Worth setting where the target's position is fixed by
              the rig: it cuts the search cost and, more importantly, stops a
              pattern elsewhere in the frame from out-correlating the real
              target.
    """
    searched, (offset_x, offset_y) = _region_slice(image, region)
    gray = searched if searched.ndim == 2 else cv2.cvtColor(searched, cv2.COLOR_BGR2GRAY)

    coarse = _match_at_scales(gray, template, coarse_scales)
    if coarse is None or coarse[3] < match_score_min:
        return None

    won_scale = coarse[4] / template.shape[1]
    step = coarse_scales[1] - coarse_scales[0]
    fine = _match_at_scales(gray, template,
                            np.linspace(won_scale - step, won_scale + step, 21))

    best = fine if (fine is not None and fine[3] >= coarse[3]) else coarse
    cx, cy, radius, score, _ = best
    return cx + offset_x, cy + offset_y, radius, score


def scale_from_target(image, template, target_mm, region=None,
                      coarse_scales=DEFAULT_SCALES,
                      match_score_min=MATCH_SCORE_MIN, name=None):
    """
    Pixels per millimetre from an image containing a target of known width.

    image     -- BGR (or grayscale) array showing the target.
    template  -- grayscale template image, cropped tightly to the target.
    target_mm -- the target's real width in millimetres. A US 1-inch quadrant
                 circle is 25.4; measure yours rather than trusting a spec
                 sheet, since printers scale.

    Returns the measurement as a dict, or None if no target was found -- not
    zero, and not a raise: an image without a visible target is a normal thing
    in a large collection, and the caller decides whether it's a problem.
    """
    started = time.perf_counter()
    found = detect_target(image, template, region=region,
                          coarse_scales=coarse_scales,
                          match_score_min=match_score_min)
    elapsed = time.perf_counter() - started
    if found is None:
        logger.warning("no scale target detected%s after %.1fs -- if it's "
                       "visible in the frame, try lowering match_score_min, "
                       "widening coarse_scales, or checking region",
                       f" in {name}" if name else "", elapsed)
        return None

    cx, cy, radius, score = found
    diameter_px = 2 * radius
    px_per_mm = diameter_px / float(target_mm)

    # The elapsed time is worth reporting: a full-resolution light-trap sheet
    # takes tens of seconds to sweep, which is the dominant cost of a scale pass
    # and not at all obvious from the outside.
    logger.info("%s: target at (%d,%d) r=%dpx match=%.3f -> %.4f px/mm (%.1fs)",
                name or "image", cx, cy, radius, score, px_per_mm, elapsed)
    if score < WEAK_MATCH_SCORE:
        logger.warning("%s: weak match (%.3f) -- this may be clutter rather "
                       "than the target, and a wrong scale is worse than none. "
                       "Check the panel, or raise match_score_min",
                       name or "image", score)

    return {"px_per_mm": float(px_per_mm), "score": float(score),
            "cx": int(cx), "cy": int(cy), "radius_px": int(radius),
            "diameter_px": int(diameter_px)}


def scale_panel(image, result):
    """
    The measurement drawn on the image: the matched circle, its centre, and the
    numbers -- the view that shows whether a plausible px/mm came from finding
    the target or from finding something else that looked like it.
    """
    panel = np.asarray(image).copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    cv2.circle(panel, (result["cx"], result["cy"]), result["radius_px"],
               (0, 255, 0), 2)
    cv2.circle(panel, (result["cx"], result["cy"]), 2, (0, 0, 255), 3)
    annotate(panel, f"{result['px_per_mm']:.4f} px/mm  match {result['score']:.3f}")
    return panel


def make_scale_row(scope, scope_value, px_per_mm, source, score=None,
                   measured_from=None):
    """
    Build one scale calibration record, ready for
    records.calibrations.save_calibrations.

    The type-specific wrapper around a generic row: it knows the
    calibration_type, it knows the parameter is called px_per_mm, and it knows
    what a valid one looks like. The record layer knows none of those and
    shouldn't -- validating a colour matrix and validating a scale have nothing
    in common but the word.

    px_per_mm must be positive. A zero or negative scale is a failed
    measurement that got written by mistake, and dividing by it later produces
    infinities in a trait table rather than an error anyone notices.
    """
    px_per_mm = float(px_per_mm)
    if not px_per_mm > 0:
        raise ValueError(
            f"px_per_mm must be positive, got {px_per_mm} for "
            f"{scope}={scope_value!r}"
        )

    return calibration_records.make_calibration_row(
        CALIBRATION_TYPE, scope, scope_value, {SCALE_COL: px_per_mm},
        source=source, score=score, measured_from=measured_from)


def declare_scale(project_path, px_per_mm, scope=ID_COL, scope_value=None,
                  measured_from=None):
    """
    Record a scale you already know, rather than one measured off an image.

    For a rig whose calibration is a fact about the equipment: a copy stand at a
    fixed height, a microscope objective, a scanner at a stated dpi. There's
    nothing to detect in those images and no reason to pretend otherwise.

    project_path -- project to record it in.
    px_per_mm    -- pixels per millimetre.
    scope        -- occurrence column this applies across; ID_COL (the default)
                    means one occurrence.
    scope_value  -- the value in that column. Required.

    A scanner's dpi converts as px_per_mm = dpi / 25.4.
    """
    if scope_value is None:
        raise ValueError(
            "declare_scale needs a scope_value -- which occurrence, session, or "
            f"device is {px_per_mm} px/mm true of?"
        )

    row = make_scale_row(scope, scope_value, px_per_mm, source="declared",
                         measured_from=measured_from)
    calibration_records.save_calibrations(project_path, [row])
    logger.info("declared %.4f px/mm for %s=%s", px_per_mm, scope, scope_value)
    return row


def scale_for_occurrences(project_path, occurrence_ids=None):
    """
    px_per_mm per occurrence, as a float Series indexed by occurrence_id, or NaN
    where nothing applies.

    The scale-shaped view of records.calibrations.resolve_for_occurrences, which
    does the work of deciding which row covers which occurrence (and in what
    order of specificity) and hands back parameter dicts. Pulling one number out
    of them is all that's left, and it's this module's job because only this
    module knows the number is called px_per_mm.
    """
    resolved = calibration_records.resolve_for_occurrences(
        project_path, CALIBRATION_TYPE, occurrence_ids=occurrence_ids)
    if resolved.empty:
        return pd.Series(dtype="float64", name=SCALE_COL)

    values = resolved.map(
        lambda parameters: parameters.get(SCALE_COL)
        if isinstance(parameters, dict) else None)
    return values.astype("float64").rename(SCALE_COL)


def pending_scope_values(project_path, scope, limit=None):
    """Scope values with no scale calibration yet -- see records.calibrations."""
    return calibration_records.pending_scope_values(
        project_path, CALIBRATION_TYPE, scope, limit=limit)


def _measured_values(project_path, scope):
    """The scope values that already have a scale, as a set of strings."""
    measured = calibration_records.load_calibrations(
        project_path, calibration_type=CALIBRATION_TYPE, scope=scope)
    return set(measured["scope_value"].astype(str))


def measure_scales(project_path, template, target_mm, scope=ID_COL, region=None,
                   match_score_min=MATCH_SCORE_MIN, subset=None, limit=None,
                   force=False, visualize=False):
    """
    Measure a scale target in each occurrence's own image and record the result.

    The per-image case: a specimen photographed with a target beside it. Where
    the target is on a separate image not in the store (a light-trap sheet the
    crops were cut from), measure that image yourself and write the row directly
    -- see the antenna_lighttraps extension.

    project_path -- project to measure and record in.
    template     -- grayscale template, cropped tightly to the target.
    target_mm    -- the target's real width in millimetres.
    scope        -- occurrence column the rows are keyed on. ID_COL by default,
                    the honest answer when every frame is measured separately.
                    Naming a grouping column records one measurement as covering
                    the whole group, true only if the rig really was fixed.
    subset       -- named subset to restrict to.
    limit        -- optional cap, for checking a template works first.
    force        -- re-measure scope values that already have a scale.
    visualize    -- save one panel per measurement under visualizations/scale/.

    Returns a summary dict (measured, skipped, failed, missed).
    """
    paths.require_project(project_path)

    calibration_records.require_scope_column(project_path, scope)
    occurrences = subset_selection.select_occurrences(project_path, subset=subset,
                                                      columns=[scope])

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
    if limit is not None:
        todo = todo[:limit]

    skipped = len(occurrences) - len(todo)
    logger.info("measuring scale on %d image(s) keyed by '%s'; %d already "
                "covered or duplicated", len(todo), scope, skipped)

    rows = []
    failed = 0
    missed = 0

    with ImageStore(project_path, readonly=True) as images:
        for occurrence_id, value in todo:
            try:
                image = images.get(occurrence_id)
                if image is None:
                    raise ValueError("no image in the image store")

                result = scale_from_target(image, template, target_mm,
                                           region=region,
                                           match_score_min=match_score_min,
                                           name=str(occurrence_id))
                if result is None:
                    missed += 1
                    continue

                if visualize:
                    save_panel(project_path, scale_panel(image, result),
                               str(occurrence_id), subdir="scale")

                rows.append(make_scale_row(
                    scope, value, result["px_per_mm"], source="target",
                    score=result["score"], measured_from=str(occurrence_id)))

            except Exception as exc:
                failed += 1
                logger.warning("scale measurement failed for %s: %s",
                               occurrence_id, exc)

    calibration_records.save_calibrations(project_path, rows)
    logger.info("scale pass complete: measured=%d skipped=%d failed=%d missed=%d",
                len(rows), skipped, failed, missed)
    return {"measured": len(rows), "skipped": skipped, "failed": failed,
            "missed": missed}
