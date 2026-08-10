import sys
import numpy as np
import cv2
from mm2tools.datahelpers.visualize import save_visualization
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# real diameter of the quadrant target
CIRCLE_DIAMETER_MM = 25.4  # 1 inch

# Quadrant-pattern acceptance threshold. Real target scores ~60+, false
# circles ~1-5, so 20 is a safe cut with wide margin. Lower if a valid target
# is ever rejected (e.g. very low contrast lighting); raise if a false circle
# ever passes.

def _quadrant_contrast(gray, cx, cy, r):
    """
    Score how 'quadrant checker' a circle's interior is. A real black/white
    quadrant target has two matching dark quarters on one diagonal and two
    matching light quarters on the other. Returns a score that is high only
    when the diagonals differ AND each diagonal is internally consistent —
    which an oversized ring around the target fails.
    """
    off = int(r * 0.45)
    s = max(2, int(r * 0.25))

    def patch_mean(qx, qy):
        p = gray[max(0, qy - s):qy + s, max(0, qx - s):qx + s]
        return float(p.mean()) if p.size else np.nan

    tl = patch_mean(cx - off, cy - off)
    tr = patch_mean(cx + off, cy - off)
    bl = patch_mean(cx - off, cy + off)
    br = patch_mean(cx + off, cy + off)
    if any(np.isnan([tl, tr, bl, br])):
        return 0.0

    diag1 = (tl + br) / 2.0    # one diagonal pair
    diag2 = (tr + bl) / 2.0    # the other
    contrast = abs(diag1 - diag2)          # diagonals should differ
    consistency = abs(tl - br) + abs(tr - bl)   # each pair should match (low = good)

    # high contrast, low within-diagonal spread → real quadrant target
    return contrast - consistency

def _detect_target_circle(bgr,
                         min_radius=250,
                         max_radius=375,
                         quadrant_contrast_min=50,
                         hough_param2=20,
                         visualize=False):
    """
    Detect the quadrant target circle. Returns (cx, cy, r, contrast) or None.
    Crops to the top-left quadrant (where the card always sits), finds all
    circle candidates, keeps those with a quadrant contrast above a threshold,
    and returns the largest of those.
    """
    h, w = bgr.shape[:2]
    x0, y0 = 0, 0
    x1, y1 = w // 2, h // 2
    crop = bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=100, param2=hough_param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None

    scored = []
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        contrast = _quadrant_contrast(gray, cx, cy, r)
        if contrast >= quadrant_contrast_min:
            scored.append((cx, cy, r, contrast, contrast * r))

    if not scored:
        return None
    scored.sort(key=lambda t: t[3], reverse=True)  # t[3] is contrast/score
    cx, cy, r, contrast, _ = scored[0]

    return cx + x0, cy + y0, r, contrast

def scale_from_image(path, visualize=False):
    """
    Recover image scale (pixels per mm) from an image using the ForensiGraph reference card in the top-left corner.
    Works by detecting the 1 in. black/white QUADRANT target circle and obtaining its pixel radius.
    Can be adjusted later for cards with target circles of other sizes.

    Ranked sources of error to consider
    1. Perspective
    2. Lens distortion
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    result = _detect_target_circle(bgr,visualize=visualize)
    if result is None:
        print("No quadrant target detected. If the card is visible, try "
              "lowering QUADRANT_CONTRAST_MIN or widening the radius range.")
        return None

    cx, cy, r, contrast = result
    diameter_px = 2 * r
    px_per_mm = diameter_px / CIRCLE_DIAMETER_MM

    print(f"target: center=({cx},{cy}) radius={r}px diameter={diameter_px}px "
          f"quadrant_contrast={contrast:.1f}")
    print(f"scale : {px_per_mm:.3f} px/mm  (circle = {CIRCLE_DIAMETER_MM} mm)")
    print(f"        {px_per_mm * 10:.2f} px/cm")

    if visualize:
        out = bgr.copy()
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)

        stem = Path(path).stem
        save_visualization(out, f"{stem}_scale_overlay.png")

    return {"cx": cx, "cy": cy, "radius_px": r, "diameter_px": diameter_px,
            "px_per_mm": px_per_mm, "quadrant_contrast": contrast}