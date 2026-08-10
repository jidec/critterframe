"""
Every named crop region, drawn on one image.

Two things to check here, and the second is the one that matters. First, that a
region lands where you think it does -- "upper_right" is upper right of the
CURRENT frame, which is not the original frame once something has already
cropped. Second, that a mask found inside a crop comes back pointing at the
right pixels of the parent image, which is the mechanism the whole parts-as-
crops approach rests on.

Needs no project and no model -- it draws its own image.

Run from the repo root:
    python scripts/simple_tests/transforms/crop_test.py
"""

import cv2
import numpy as np

import critterframe as cf
from critterframe.recipes import Segment
from critterframe.transforms.crop import REGIONS

# A frame with a numbered blob in each quadrant, so a misplaced crop is obvious.
image = np.full((300, 400, 3), 30, np.uint8)
blobs = {"upper_left": (100, 75), "upper_right": (300, 75),
         "lower_left": (100, 225), "lower_right": (300, 225)}
for label, (x, y) in blobs.items():
    cv2.circle(image, (x, y), 35, (200, 200, 200), -1)
    cv2.putText(image, label[0].upper() + label.split("_")[1][0].upper(),
                (x - 15, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 100

print(f"frame is {image.shape[1]}x{image.shape[0]}; blob centres: {blobs}\n")

print("== named regions ==")
for region in REGIONS:
    segment = Segment(image, mask=mask, occurrence_id="test")
    cropped, info = cf.crop(region=region)(segment)
    print(f"  {region:<12} -> box ({info['x']:3d},{info['y']:3d}) "
          f"{info['width']}x{info['height']}")

print("\n== a mask found inside a crop, restored to the parent frame ==")
for region, (x, y) in blobs.items():
    segment = Segment(image, mask=None, occurrence_id="test")
    cropped, _info = cf.crop(region=region)(segment)

    # pretend a segmenter ran on the cropped frame and found the blob there
    found = cv2.cvtColor(np.asarray(cropped.image), cv2.COLOR_BGR2GRAY) > 100
    cropped = cropped.replace(mask=found)

    restored = cropped.mask_in_original_coordinates()
    ys, xs = np.nonzero(restored)
    print(f"  {region:<12} working frame {cropped.shape[1]}x{cropped.shape[0]:<4} "
          f"-> restored to {restored.shape} centred "
          f"({xs.mean():.0f}, {ys.mean():.0f}), expected ({x}, {y})")

print("\n== chained spatial transforms still invert ==")
segment = Segment(image, mask=mask, occurrence_id="test")
for operation in (cf.crop(region="lower_right"), cf.rotate(25), cf.resize(scale=1.5)):
    segment, info = operation(segment)
    print(f"  after {operation.name:<8} frame is "
          f"{segment.shape[1]}x{segment.shape[0]}")

restored = segment.mask_in_original_coordinates()
print(f"  restored shape {restored.shape} (expect {image.shape[:2]}) -- one "
      "inversion undoes the whole chain, in any order")
