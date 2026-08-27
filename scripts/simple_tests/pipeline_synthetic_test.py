"""
End-to-end pipeline over synthetic images -- the one script here that needs no
real data, no credentials, no GPU, and no network.

Builds a throwaway project of drawn "specimens", segments them with a trivial
threshold model, measures them, exports, then resegments and does it again.
Worth running after any change to the core machinery: it exercises ingest, the
image store, recipe hashing, repeat-awareness, coordinate inversion, the mask
table, the metric log, metric staleness after a resegmentation, and export in
one pass, and it finishes in a second.

Run from the repo root:
    python scripts/simple_tests/pipeline_synthetic_test.py
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd

import critterframe as cf
from critterframe.records import calibrations as calibration_records
from critterframe.records.metrics import load_metrics

# The specimens and the stand-in segmenter are shared with the test suite rather
# than written out twice. This script shows a person what the pipeline did and
# the suite asserts what it computed, and both have to be looking at the same
# thing for either to mean anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from helpers.models import ThresholdModel          # noqa: E402
from helpers.synthetic import write_specimens      # noqa: E402

logging.basicConfig(level=logging.WARNING)

WORKSPACE = os.path.join(tempfile.gettempdir(), "critterframe_synthetic_test")
PROJECT_PATH = os.path.join(WORKSPACE, "project")
IMAGE_DIR = os.path.join(WORKSPACE, "images")

shutil.rmtree(WORKSPACE, ignore_errors=True)
os.makedirs(IMAGE_DIR)

# Tilted ellipses with thin "legs" -- enough shape for orientation to have an
# answer and for appendage removal to have something to remove. Drawn by
# tests/helpers/synthetic.py; open it to see exactly what a "specimen" is here.
write_specimens(Path(IMAGE_DIR), count=8)

# ThresholdModel is the smallest thing meeting the segmenter contract: predict()
# returning a mask, a score, and diagnostics, plus identity() so it reaches the
# recipe hash. A real model differs only in what happens inside predict(). Its
# `erode` parameter stands in for "a different segmenter that finds a slightly
# different organism", which is what the resegmentation section below needs --
# and it's in identity(), so the two configurations hash differently.


print("== ingest ==")
# A `device` column so the scale section below has a real grouping to key on:
# two imaginary camera rigs, alternating specimens.
devices = pd.DataFrame({
    "occurrence_id": [f"specimen{index}" for index in range(8)],
    "device": ["boxA" if index % 2 == 0 else "boxB" for index in range(8)],
})
print(cf.ingest_images(PROJECT_PATH, IMAGE_DIR, metadata=devices))

print("\n== segment ==")
steps = [cf.segment(ThresholdModel())]
print("first run :", cf.run_segments(PROJECT_PATH, steps=steps))
print("second run:", cf.run_segments(PROJECT_PATH, steps=steps),
      "  <- expect processed=0, skipped=8 (repeat-aware)")
print("forced    :", cf.run_segments(PROJECT_PATH, steps=steps, force=True),
      "  <- expect processed=8")

print("\n== metrics ==")
metrics = [cf.body_length(), cf.max_width(), cf.mask_area(name="area_px", unit="px2"),
           cf.mean_lightness(), cf.blur_variance(), cf.bilateral_asymmetry(),
           cf.edge_fraction()]


def measure(**kwargs):
    """The one metric recipe this script reruns, so every run below is identical."""
    return cf.run_metrics(PROJECT_PATH, run_name="traits",
                          transforms=[cf.remove_appendages(), cf.orient()],
                          metrics=metrics, **kwargs)


print("first run :", measure())
print("second run:", measure(), "  <- expect processed=0, skipped=8")

print("\n== export ==")
traits = cf.export_metrics(PROJECT_PATH, os.path.join(PROJECT_PATH, "traits.csv"))
print(traits.to_string(index=False))
print("\nNOTE: some specimens come back with length and width swapped, and that")
print("is correct behaviour, not a bug. orient() picks the body axis by")
print("ASYMMETRY -- a real organism's head end and tail end differ, a plain")
print("ellipse's don't -- so on a perfectly symmetric synthetic shape the")
print("choice is noise. compute_orientation reports it as unreliable via its")
print("eigenvalue ratio; run transforms/orient_test.py on real crops to see it")
print("working on shapes that actually have an asymmetry to find.")
print("\nunits:", cf.export_units(PROJECT_PATH))

print("\n== filtered export ==")
filtered = cf.export_metrics(PROJECT_PATH,
                             filters={"traits__organism__body_length": (">", 80)})
print(f"{len(filtered)} of {len(traits)} occurrences pass body_length > 80 "
      "(nothing was deleted -- the rest are still in the project)")

print("\n== scale: pixels into millimetres ==")
# boxA's rig is calibrated; boxB's isn't. specimen0 was shot with a target in
# frame, so it has its own measurement, which must beat its device's.
cf.declare_scale(PROJECT_PATH, 4.0, scope="device", scope_value="boxA")
cf.declare_scale(PROJECT_PATH, 8.0, scope="occurrence_id", scope_value="specimen0")

print(cf.scale_for_occurrences(PROJECT_PATH).to_string())
print("  <- specimen0 gets 8.0, not boxA's 4.0: a scale measured on one frame")
print("     is more specific than one measured for the rig it was shot on.")
print("     The boxB specimens have no calibration and stay NaN.")

mm = cf.export_metrics(PROJECT_PATH, units="mm")
print("\n" + mm[["occurrence_id", "px_per_mm",
                 "traits__organism__body_length_mm",
                 "traits__organism__area_px_mm2",
                 "traits__organism__mean_lightness"]].to_string(index=False))
print("  <- lengths divided by px_per_mm and areas by its square, each renamed")
print("     with the unit it now carries. mean_lightness is a fraction, so it")
print("     is left alone. Uncalibrated occurrences are NaN, never raw pixels")
print("     sitting in a column labelled mm.")

# The table is not scale-shaped. A calibration type it knows nothing about,
# whose parameters are a matrix rather than a number, stores and resolves the
# same way -- which is the whole point of keying on (type, scope, scope_value)
# and leaving the payload opaque.
calibration_records.save_calibrations(PROJECT_PATH, [
    calibration_records.make_calibration_row(
        "color", "device", "boxA",
        {"method": "rgb_affine", "matrix": [[1.02, 0, 0], [0, 0.98, 0], [0, 0, 1.05]],
         "offset": [0.01, 0.0, -0.01], "illuminant": "D65"},
        source="color_checker", score=0.94),
])
resolved = calibration_records.resolve_for_occurrences(PROJECT_PATH, "color")
print("\na 'color' calibration on the same device resolves alongside the scale:")
print(f"  specimen0 -> {resolved['specimen0']}")
print(f"  specimen1 -> {resolved['specimen1']}  <- boxB, uncalibrated")
print("  <- same scope machinery, arbitrary parameters, no schema change. The")
print("     scale row for boxA is untouched: the key includes the type.")

print("\n== resegment, and what that does to numbers measured off the old mask ==")
LENGTH = "traits__organism__body_length"
first_id = traits["occurrence_id"].iloc[0]
before = traits.set_index("occurrence_id")[LENGTH][first_id]

print("resegment :", cf.run_segments(PROJECT_PATH, steps=[cf.segment(ThresholdModel(erode=2))]))
stale = cf.export_metrics(PROJECT_PATH)
print(f"export now: {len(stale)} of {len(traits)} occurrences have a current value")
print("  <- expect 0. Every stored value was measured off the mask that just got")
print("     replaced, so none of them describes what the project now holds. They")
print("     are not gone -- see the long table below.")

print("remeasure :", measure(),
      "  <- expect processed=8, skipped=0. The recipe has run over these")
print("     occurrences before, but not over these masks, so there is no cached")
print("     work to reuse and force= isn't needed to get it redone.")

after = cf.export_metrics(PROJECT_PATH).set_index("occurrence_id")[LENGTH][first_id]
print(f"\n{first_id} body_length: {before:.1f} -> {after:.1f} "
      "(a tighter mask is a shorter body)")

history = load_metrics(PROJECT_PATH, metric_names=["body_length"])
history = history[history["occurrence_id"] == first_id]
print("\nboth values are still on record, distinguished by the mask they came from:")
print(history[["metric_id", "value", "recipe_hash", "source_mask_hash",
               "run_name", "run_created_at"]].to_string(index=False))
print("same metric recipe_hash, different source_mask_hash -- which is exactly")
print("what the rerun and the export each keyed their decision on.")

print("\n== a part derived from another part follows it ==")
# "core" is cut out of the organism mask: remove_background() blanks everything
# outside it, so what the core segmenter can find is bounded by the organism
# mask it started from -- which is what makes the organism a real dependency
# rather than a formality.
def segment_core(**kwargs):
    return cf.run_segments(PROJECT_PATH, run_name="core", part="core",
                           from_part="organism",
                           shared_steps=[cf.remove_background()],
                           steps=[cf.segment(ThresholdModel(erode=1))], **kwargs)


def measure_core(**kwargs):
    return cf.run_metrics(PROJECT_PATH, run_name="core_traits", part="core",
                          metrics=[cf.mask_area(name="area_px", unit="px2")],
                          **kwargs)


print("segment core :", segment_core())
print("again        :", segment_core(), "  <- expect processed=0, skipped=8")
print("measure core :", measure_core())
print("again        :", measure_core(), "  <- expect processed=0, skipped=8")

core_before = cf.export_metrics(PROJECT_PATH, parts=["core"]).set_index("occurrence_id")
print("\nresegment the organism it was cut out of, and nothing about the core")
print("recipe changes -- but everything under it is stale:")
print("resegment    :", cf.run_segments(PROJECT_PATH, steps=[cf.segment(ThresholdModel())]))
print("segment core :", segment_core(),
      "  <- expect processed=8: same recipe hash, different organism mask")
print("measure core :", measure_core(), "  <- expect processed=8")

core_after = cf.export_metrics(PROJECT_PATH, parts=["core"]).set_index("occurrence_id")
AREA = "core_traits__core__area_px"
print(f"\n{first_id} core area: {core_before[AREA][first_id]:.0f} -> "
      f"{core_after[AREA][first_id]:.0f} (back to the un-eroded organism, so bigger)")

print("\n== pipeline visualization: one sheet per run, from a sample ==")
# force=True because a grid can only show work that actually happened, and by
# now every occurrence is already segmented and measured.
print("segment   :", cf.run_segments(PROJECT_PATH, steps=[cf.segment(ThresholdModel())],
                                     force=True, visualize=6))
print("explicit  :", measure(force=True, visualize=["specimen0", "specimen3"]),
      "  <- named occurrences instead of a sample")
print("metrics   :", measure(force=True, visualize=6),
      "  <- same run name and recipe, so this REPLACES the grid above")
print("derived   :", segment_core(force=True, visualize=4),
      "  <- a part's grid carries the shared steps that fed it")

for grid in sorted(Path(PROJECT_PATH, "visualizations", "pipeline").glob("*.jpg")):
    height, width = cv2.imread(str(grid)).shape[:2]
    print(f"  {grid.name}  {width}x{height}")
print("  <- one file per run+recipe: segments are a single-stage image grid,")
print("     the metric run is a row per specimen x a column per stage")
print("     (remove_appendages | orient | measured).")

print("\n== products: one file per occurrence-part, for figures and R ==")
plates = cf.render_segments(
    PROJECT_PATH, "oriented_bodies",
    transforms=[cf.remove_background(), cf.orient(), cf.crop_to_mask(pad=0.15)])
print(plates)
print("again     :", cf.render_segments(
    PROJECT_PATH, "oriented_bodies",
    transforms=[cf.remove_background(), cf.orient(), cf.crop_to_mask(pad=0.15)]),
    "  <- expect rendered=0, skipped=8: same recipe, same folder, files there")

files = sorted(p.name for p in plates["directory"].glob("*.png"))
print(f"  {plates['directory'].name}/  ->  {', '.join(files[:4])}, ...")

both = cf.render_segments(PROJECT_PATH, "parts_plate",
                          transforms=[cf.remove_background(), cf.crop_to_mask()],
                          parts=["organism", "core"])
print("  two parts ->", ", ".join(sorted(p.name for p in
                                         both["directory"].glob("*.png"))[:3]),
      "...  <- part qualifies the filename when a render covers several")

print("\n== validation: comparing against a reference ==")
# A second, deliberately different segmenter standing in for the human whose
# corrections normally fill the reference table -- correct_mask() blocks on a
# person, and what validation reads is the TABLE, not who wrote it.
print("reference masks  :", cf.run_segments(
    PROJECT_PATH, run_name="stand_in_reference",
    steps=[cf.segment(ThresholdModel(erode=3))], reference=True))
print(f"mask agreement   : mean iou "
      f"{cf.validate_masks(PROJECT_PATH)['iou'].mean():.3f}"
      "  <- a 3px erosion, so expect high but not perfect")

# The same measurement method over the other mask table, so what differs
# between the two runs is the masks and nothing else.
print("reference metrics:", cf.run_metrics(
    PROJECT_PATH, run_name="reference_traits",
    transforms=[cf.remove_appendages(), cf.orient()],
    metrics=[cf.body_length(), cf.mask_area(name="area_px", unit="px2"),
             cf.mask_area()],
    reference=True))

print("\nsame name on both sides -- every metric the two runs share:")
print(cf.compare_metrics(PROJECT_PATH, "traits", "reference_traits")
      .to_string(index=False))

print("\ndifferent names, paired explicitly. Nothing here is a human label, but")
print("the shape is the one that grades a trait against one -- the reference")
print("run's plain mask_area against the traits run's area_px:")
print(cf.compare_metrics(PROJECT_PATH, "traits", "reference_traits",
                         metric_names={"area_px": "mask_area"})
      .to_string(index=False))
print("  <- same numbers as the area_px row above, reached by pairing two")
print("     differently-named columns. A dict-valued metric's key is named the")
print("     same way: click_two_points__length_px.")

print("\n== summary ==")
cf.print_summary(PROJECT_PATH)
print(f"\nproject left at {PROJECT_PATH} for poking at")
