"""
Look at a project's occurrence table.

The first thing to check after an ingest: are the ids what you expect, is
image_url populated, and did the source's other columns survive? They should --
normalization renames two columns and keeps everything else.

Run from the repo root:
    python scripts/simple_tests/records/occurrences_test.py
"""

import logging

from critterframe.project import summarize
from critterframe.records.occurrences import ID_COL, IMAGE_URL_COL, load_occurrences

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

occurrences = load_occurrences(PROJECT_PATH)
print(f"{len(occurrences)} occurrences, {len(occurrences.columns)} columns")
print(f"columns: {sorted(occurrences.columns)}")

print(f"\nid dtype: {occurrences[ID_COL].dtype} (expect object -- ids are strings "
      "everywhere, so a numeric id from the source can't fail to match an LMDB "
      "key or a sqlite column later)")
print(f"duplicate ids: {int(occurrences[ID_COL].duplicated().sum())} (expect 0 -- "
      "one occurrence is one organism in one image)")

if IMAGE_URL_COL in occurrences.columns:
    missing = int(occurrences[IMAGE_URL_COL].isna().sum())
    print(f"occurrences with no image URL: {missing} -- these can't be downloaded")

print("\nfirst rows:")
print(occurrences.head(5).to_string())

print()
summarize.print_summary(PROJECT_PATH)
