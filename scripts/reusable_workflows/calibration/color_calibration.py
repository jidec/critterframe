"""
Colour that can be compared across photos taken under lighting nobody
controlled.

There is no colour calibration record yet: calibrations/ only has scale, and a
colour card measured per image (the counterpart of measure_scales) is not
written. What exists today is correction computed from each image itself, and
this workflow uses that:
- white_balanced_color -- grey-world white balance: assume the whole frame
  averages to grey, correct each channel so it does, then take the organism's
  mean colour. Crude, and wrong for a frame that is mostly one colour (a
  specimen filling a green leaf), but it needs no reference object.
- background_color -- the colour of everything outside the mask, and the
  organism's lightness contrast against it. Records the conditions the
  correction had to work with, and flags low-contrast images whose masks and
  colours are least trustworthy.

Raw mean colour is measured beside both, so the effect of the correction can
be checked rather than assumed.
"""

import logging

import critterframe as cf

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
PART = "organism"
RUN_NAME = "color"

cf.run_metrics(
    PROJECT_PATH, run_name=RUN_NAME, part=PART,
    transforms=[cf.remove_appendages()],
    metrics=[
        cf.mean_color(name="raw_color"),
        cf.white_balanced_color(),
        cf.background_color(),
    ],
)

# Low contrast against the background means the mask most likely took some
# substrate with it, so its colours are contaminated. A starting cutoff to
# check against the grids, not a calibrated one; see
# validation/validate_filters.py for calibrating cutoffs against labels.
table = cf.export_metrics(PROJECT_PATH, run_names=[RUN_NAME], path=False)
contrast = table[f"{RUN_NAME}__{PART}__background_color__contrast"].abs()
logging.info("%d occurrences; %d with |contrast| < 0.05", len(table), (contrast < 0.05).sum())
