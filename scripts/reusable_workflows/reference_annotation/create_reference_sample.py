"""
Step 1 of 3 of building a reference set: choose the occurrences a human will
look at.

The sample is a subset, grown rather than redrawn: raising TARGET_SIZE later
adds only the shortfall and keeps every occurrence already in it, so screening
and reference masks already made are never orphaned by a bigger sample.

Next: annotate_usability.py screens this sample; annotate_reference_masks.py
then draws masks on the usable part of it.
"""

import logging

import critterframe as cf
from critterframe.project.subsets import select_occurrences

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

CANDIDATES_SUBSET = "reference_candidates"
TARGET_SIZE = 200

# Spread the sample across a grouping column so common groups can't crowd out
# rare ones -- e.g. "species", "device", "institutionCode". None samples the
# whole project uniformly.
STRATIFY_COL = "species"

if STRATIFY_COL is None:
    ids = cf.grow_subset(PROJECT_PATH, CANDIDATES_SUBSET, target_size=TARGET_SIZE)
else:
    # sample_per_group is deterministic: smallest group first, each capped at
    # its own size. Passing the stratified draw as candidate_ids and the same
    # target size makes grow_subset take all of it on the first call, and keep
    # it on every later one.
    occurrences = select_occurrences(PROJECT_PATH, columns=[STRATIFY_COL])
    stratified = cf.sample_per_group(occurrences, STRATIFY_COL, TARGET_SIZE)
    ids = cf.grow_subset(PROJECT_PATH, CANDIDATES_SUBSET, target_size=TARGET_SIZE,
                         candidate_ids=stratified)

logging.info("'%s' holds %d occurrences", CANDIDATES_SUBSET, len(ids))
