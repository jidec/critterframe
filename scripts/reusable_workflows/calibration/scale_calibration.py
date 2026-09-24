"""
Pixels per millimetre for every occurrence, so pixel traits can be exported in
millimetres.

Pick the METHOD matching how the images were taken:
- "target"   -- a scale target of known width (a printed circle, a ruler's
                mark) is inside every photo; it's found by template matching,
                one measurement per image.
- "declared" -- a fixed rig (copy stand, microscope objective, scanner dpi);
                the scale is a known fact, recorded once for everything
                sharing a column value.
- "clicked"  -- the scale bar is in a separate scene image (a light-trap
                sheet); click its two ends once, and that scale covers every
                occurrence.

Measuring always stays in pixels; conversion happens at export (units="mm").
A corrected calibration therefore costs one re-export, never a recompute.
"""

import logging

import cv2

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
METHOD = "target"

if METHOD == "target":
    # Crop one clean example of the target tightly and save it as grayscale.
    # Measure its real width yourself: printers rescale.
    template = cv2.imread("source/scale_target.png", cv2.IMREAD_GRAYSCALE)
    # max_new=10 tries the template on ten images first; open the grid of
    # weakest matches it writes, then remove max_new to measure the rest.
    # A weak match is recorded but warned about, since clutter can outscore a
    # missing target.
    summary = cf.measure_scales(PROJECT_PATH, template, target_mm=25.4, max_new=10)
    logging.info("measured %s, no target found in %s", summary["processed"], summary["missed"])

elif METHOD == "declared":
    # One row per rig. scope names an occurrence column; every occurrence
    # sharing scope_value gets this scale. A scanner: px_per_mm = dpi / 25.4.
    cf.declare_scale(PROJECT_PATH, px_per_mm=600 / 25.4,
                     scope="device", scope_value="scanner_1")

elif METHOD == "clicked":
    scene = cv2.imread("source/trap_sheet_with_scale_bar.jpg")
    # target_mm=None prompts for the length after the two clicks.
    cf.measure_scale_by_hand(PROJECT_PATH, scene, target_mm=10.0)

# Coverage check: an occurrence without a scale exports blank mm columns.
scales = cf.scale_for_occurrences(PROJECT_PATH)
logging.info("px/mm known for %d of %d occurrences (median %.2f)",
             scales.notna().sum(), len(scales), scales.median())

# Export in millimetres; each converted column's calibration goes in the
# export manifest.
cf.export_metrics(PROJECT_PATH, units="mm")
