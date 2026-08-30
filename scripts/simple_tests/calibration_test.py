"""
Scale recovery from a target of known size -- no project, no network.

Draws a synthetic "sheet" with a circular target of a known pixel diameter,
crops that circle out as the template, and checks the measurement comes back at
the scale it was drawn at. Because the drawing is exact, the expected answer is
known to the pixel, which a real photograph could never give you.

Run from the repo root:
    python scripts/simple_tests/calibration_test.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2

from critterframe.calibrations.scale import scale_from_target, scale_panel

# The target sheet is shared with the test suite rather than drawn twice --
# this is where the drawing was originally written, later lifted into
# tests/helpers/synthetic.py so the suite could measure the same sheet. This
# script imports the lifted version rather than keeping its own copy, so
# there's exactly one drawing to keep in sync, not two.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from helpers.synthetic import TARGET_DIAMETER_PX, TARGET_MM, draw_target_sheet  # noqa: E402

logging.basicConfig(level=logging.INFO)

OUT_DIR = os.path.join(tempfile.gettempdir(), "critterframe_calibration_test")
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED = TARGET_DIAMETER_PX / TARGET_MM

sheet, template, _expected = draw_target_sheet()
cv2.imwrite(os.path.join(OUT_DIR, "template.png"), template)

print("== a target drawn at a known size ==")
print(f"  drawn {TARGET_DIAMETER_PX}px across, {TARGET_MM}mm wide -> {EXPECTED:.4f} px/mm expected")

result = scale_from_target(sheet, template, TARGET_MM, region=(0, 0, 0.5, 0.5),
                           name="synthetic sheet")
print(f"  measured {result['px_per_mm']:.4f} px/mm  "
      f"(diameter {result['diameter_px']}px, match {result['score']:.3f})")
assert abs(result["px_per_mm"] - EXPECTED) < 1 / TARGET_MM, result
print("  <- within one pixel of the drawn diameter")

cv2.imwrite(os.path.join(OUT_DIR, "measured.png"), scale_panel(sheet, result))

print("\n== the same sheet at half resolution ==")
half = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
half_result = scale_from_target(half, template, TARGET_MM, region=(0, 0, 0.5, 0.5))
print(f"  measured {half_result['px_per_mm']:.4f} px/mm  <- expect about half "
      f"of {EXPECTED:.4f}, since a pixel now covers twice as much card")
assert abs(half_result["px_per_mm"] - EXPECTED / 2) < 2 / TARGET_MM

print("\n== searching where the target isn't: the failure mode that matters ==")
false_match = scale_from_target(sheet, template, TARGET_MM,
                                region=(0.5, 0.5, 1.0, 1.0),
                                name="bottom-right only")
print(f"  match {false_match['score']:.3f} -> {false_match['px_per_mm']:.4f} px/mm")
print("  <- WRONG, and plausible-looking: with no target in the search region,")
print("     clutter out-correlates nothing and wins. The true answer is")
print(f"     {EXPECTED:.4f}. This is why every scale row stores its score, and")
print("     why a weak match logs a warning instead of passing quietly.")

strict = scale_from_target(sheet, template, TARGET_MM, region=(0.5, 0.5, 1.0, 1.0),
                           match_score_min=0.6, name="bottom-right, strict")
print(f"\n  with match_score_min=0.6: {strict}")
print("  <- None, which is the right answer. Nothing to measure is a normal")
print("     outcome and must not be filled in with a guess.")
assert strict is None

print(f"\nwritten to {OUT_DIR}")
