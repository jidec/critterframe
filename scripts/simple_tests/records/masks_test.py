"""
Read the mask table and overlay one mask on its image.

The check that a mask is where it should be. Masks are stored RLE-encoded in
ORIGINAL image coordinates, so a mask that decodes to the wrong shape, or that
looks fine alone but sits in the wrong place, shows up here and essentially
nowhere else.

Needs a project with a segmentation run behind it.

Run from the repo root:
    python scripts/simple_tests/records/masks_test.py
"""

import logging

import cv2

from critterframe.records import masks as mask_records
from critterframe.storage.imagestore import ImageStore
from critterframe.visualization.panels import overlay_mask

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

print("parts with masks:", mask_records.parts_present(PROJECT_PATH))
print("parts with reference masks:",
      mask_records.parts_present(PROJECT_PATH, reference=True))

table = mask_records.load_masks(PROJECT_PATH)
if table.empty:
    print("no masks yet -- run run_segments() first")
else:
    print(f"\n{len(table)} mask row(s)")
    print(table[["occurrence_id", "part", "area", "score", "recipe_hash",
                 "from_part"]].head(10).to_string(index=False))

    row = table.iloc[0]
    mask = mask_records.decode_mask(row)
    with ImageStore(PROJECT_PATH, readonly=True) as store:
        image = store.get(row["occurrence_id"])

    print(f"\n{row['occurrence_id']} part '{row['part']}': mask {mask.shape}, "
          f"image {image.shape} -- these must match, since a mask is always "
          "stored in its original image's coordinates whatever the recipe did "
          "on the way to producing it")

    cv2.imshow(f"{row['occurrence_id']} / {row['part']}", overlay_mask(image, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
