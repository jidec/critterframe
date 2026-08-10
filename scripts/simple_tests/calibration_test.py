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
import tempfile

import cv2
import numpy as np

from critterframe.calibrations.scale import scale_from_target, scale_panel

logging.basicConfig(level=logging.INFO)

OUT_DIR = os.path.join(tempfile.gettempdir(), "critterframe_calibration_test")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MM = 25.4          # a 1-inch card target, as on a MothBox reference card
DIAMETER_PX = 120         # what we draw it at
EXPECTED = DIAMETER_PX / TARGET_MM


def draw_sheet(diameter_px=DIAMETER_PX, centre=(150, 130), size=(700, 900)):
    """A dark sheet with a quadrant-style target near the top-left, plus clutter."""
    sheet = np.full((size[0], size[1], 3), 60, np.uint8)

    # Clutter that mustn't out-correlate the target: specimen-ish blobs.
    rng = np.random.default_rng(0)
    for _ in range(25):
        x, y = rng.integers(200, size[1] - 20), rng.integers(200, size[0] - 20)
        cv2.ellipse(sheet, (int(x), int(y)), (14, 7), int(rng.integers(0, 180)),
                    0, 360, (170, 160, 140), -1)

    radius = diameter_px // 2
    cv2.circle(sheet, centre, radius, (255, 255, 255), -1)
    # Quadrant fill: two opposite quarters black, like a real calibration target.
    cv2.ellipse(sheet, centre, (radius, radius), 0, 0, 90, (0, 0, 0), -1)
    cv2.ellipse(sheet, centre, (radius, radius), 0, 180, 270, (0, 0, 0), -1)
    return sheet, centre, radius


sheet, centre, radius = draw_sheet()

# The template is the target cropped tightly to its outer edge -- the thing the
# module docstring insists on, since the matched width IS the measurement.
template = cv2.cvtColor(
    sheet[centre[1] - radius:centre[1] + radius,
          centre[0] - radius:centre[0] + radius], cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUT_DIR, "template.png"), template)

print("== a target drawn at a known size ==")
print(f"  drawn {DIAMETER_PX}px across, {TARGET_MM}mm wide -> {EXPECTED:.4f} px/mm expected")

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
