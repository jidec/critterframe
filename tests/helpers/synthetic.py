"""
Drawn images with known answers.

Synthetic rather than photographed, for one reason: the answer is known exactly.
A drawn ellipse 120 px across IS 120 px across, so `body_length` can be asserted
to the pixel instead of to a range somebody eyeballed once. Nothing here is
meant to look like an organism; it is meant to have measurable properties.

Two families:

  draw_specimen()     -- a tilted ellipse with thin legs. The legs are not
                         decoration: they are what remove_appendages() has to
                         remove and what makes a mask's outline differ from its
                         body, so a test of either has something real to work on.
  draw_target_sheet() -- a calibration card: a quadrant-filled circle of a known
                         pixel diameter on a cluttered sheet. The clutter is
                         deliberate too -- it is what a weak template match
                         latches onto when the target isn't in the search region.

Both are lifted from the smoke scripts they were first written in
(`scripts/simple_tests/pipeline_synthetic_test.py`,
`scripts/simple_tests/calibration_test.py`) so the scripts and the suite draw
identical specimens.
"""

import cv2
import numpy as np

# A drawn specimen's body, in pixels. Exposed so a test can assert against the
# number it drew with rather than restating a literal.
BODY_AXES = (20, 60)
BODY_CENTRE = (140, 110)
SPECIMEN_SIZE = (220, 280)          # (height, width)
BACKGROUND = 40
FOREGROUND = (200, 180, 150)

TARGET_MM = 25.4                    # a 1-inch card target, as on a MothBox card
TARGET_DIAMETER_PX = 120


def draw_specimen(index=0, size=SPECIMEN_SIZE, legs=True):
    """
    One drawn specimen: a tilted ellipse with two thin legs, on a dark ground.

    index -- rotates the body by 12 degrees per step, so a set of specimens
             differs in orientation while staying identical in area. That
             separates "the metric found the body" from "the metric found the
             bounding box", which a set of axis-aligned shapes cannot.
    legs  -- draw the appendages. False gives a clean ellipse for tests where an
             exact area or a symmetric shape is the subject.
    """
    image = np.full((size[0], size[1], 3), BACKGROUND, np.uint8)
    cv2.ellipse(image, BODY_CENTRE, BODY_AXES, 15 + index * 12, 0, 360,
                FOREGROUND, -1)
    if legs:
        for dx in (-45, 45):
            cv2.line(image, BODY_CENTRE, (BODY_CENTRE[0] + dx, 145),
                     FOREGROUND, 1)
    return image


def write_specimens(directory, count=8, legs=True):
    """
    Draw `count` specimens as PNGs under `directory`, returning their ids.

    PNG so nothing a test asserts on is a JPEG artifact. Ids are the filename
    stems, which is what ingest_images() keys on.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ids = []
    for index in range(count):
        occurrence_id = f"specimen{index}"
        cv2.imwrite(str(directory / f"{occurrence_id}.png"),
                    draw_specimen(index, legs=legs))
        ids.append(occurrence_id)
    return ids


def specimen_metadata(occurrence_ids):
    """
    A metadata frame for drawn specimens: an alternating `device` and a
    `species` per specimen.

    Both columns exist to be grouped on -- device is a calibration scope and a
    split's group column; species is a stratification key and a class folder --
    so the fixtures that need a grouping don't each invent one.
    """
    import pandas as pd

    return pd.DataFrame({
        "occurrence_id": list(occurrence_ids),
        "device": ["boxA" if index % 2 == 0 else "boxB"
                   for index in range(len(occurrence_ids))],
        "species": ["Anax junius" if index % 3 else "Libellula lydia"
                    for index in range(len(occurrence_ids))],
    })


def draw_target_sheet(diameter_px=TARGET_DIAMETER_PX, centre=(150, 130),
                      size=(700, 900), clutter=25):
    """
    A dark sheet with a quadrant-filled circular target near the top-left, plus
    specimen-shaped clutter.

    Returns (sheet, template, expected_px_per_mm). The template is the target
    cropped tightly to its outer edge, which is what the scale module insists on
    -- the matched width IS the measurement, so a template with margin measures
    the margin too.

    clutter -- how many blobs to scatter outside the target's quadrant. They are
               what a search region containing no target correlates against, and
               a test of the weak-match warning needs them present.
    """
    sheet = np.full((size[0], size[1], 3), 60, np.uint8)

    rng = np.random.default_rng(0)
    for _ in range(clutter):
        x, y = rng.integers(200, size[1] - 20), rng.integers(200, size[0] - 20)
        cv2.ellipse(sheet, (int(x), int(y)), (14, 7), int(rng.integers(0, 180)),
                    0, 360, (170, 160, 140), -1)

    radius = diameter_px // 2
    cv2.circle(sheet, centre, radius, (255, 255, 255), -1)
    cv2.ellipse(sheet, centre, (radius, radius), 0, 0, 90, (0, 0, 0), -1)
    cv2.ellipse(sheet, centre, (radius, radius), 0, 180, 270, (0, 0, 0), -1)

    template = cv2.cvtColor(
        sheet[centre[1] - radius:centre[1] + radius,
              centre[0] - radius:centre[0] + radius], cv2.COLOR_BGR2GRAY)
    return sheet, template, diameter_px / TARGET_MM


def blob_mask(shape=(200, 300), centre=(220, 80), axes=(30, 18), angle=0):
    """
    A boolean mask with one filled ellipse, for coordinate-inversion tests.

    Off-centre on purpose: a mask centred in its frame survives a transposed or
    mirrored affine unchanged, so it cannot tell a correct inversion from a
    lucky one.
    """
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 1, -1)
    return mask.astype(bool)
