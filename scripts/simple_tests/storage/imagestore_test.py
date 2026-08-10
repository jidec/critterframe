"""
Open a project's image store and display one image.

Confirms images actually made it in and decode back out -- the first thing to
check when a segmentation run reports "no image in the image store" for every
occurrence.

Needs a project with images ingested or downloaded. Point PROJECT_PATH at one.

Run from the repo root:
    python scripts/simple_tests/storage/imagestore_test.py
"""

import logging

import cv2

from critterframe.storage.imagestore import ImageStore

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

with ImageStore(PROJECT_PATH, readonly=True) as store:
    occurrence_ids = store.keys()
    print(f"{len(occurrence_ids)} image(s) in {PROJECT_PATH}")

    if not occurrence_ids:
        print("nothing stored yet -- run ingest_images() or download_images() first")
    else:
        occurrence_id = occurrence_ids[0]
        image = store.get(occurrence_id)
        print(f"showing {occurrence_id}: shape {image.shape}, dtype {image.dtype}")

        cv2.imshow(f"occurrence {occurrence_id}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
