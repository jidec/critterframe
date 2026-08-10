"""
Occurrence ingest, and what drop= does -- no project, no credentials, no network.

Writes a small CSV shaped like an Antenna export, ingests it with the rule that
extension uses, and checks the two halves of the contract: the occurrence table
holds only organisms, and the archived import still holds everything the source
sent.

Run from the repo root:
    python scripts/simple_tests/ingest_test.py
"""

import logging
import os
import shutil
import tempfile

import pandas as pd

import critterframe as cf
from critterframe.extensions.antenna_lighttraps.ingest import (
    NON_ORGANISM_DETERMINATIONS,
)

logging.basicConfig(level=logging.INFO)

WORKSPACE = os.path.join(tempfile.gettempdir(), "critterframe_ingest_test")
PROJECT_PATH = os.path.join(WORKSPACE, "project")
SOURCE_CSV = os.path.join(WORKSPACE, "export.csv")

shutil.rmtree(WORKSPACE, ignore_errors=True)
os.makedirs(WORKSPACE)

# One row per case worth distinguishing: two real determinations, one the source
# says isn't a moth, and one it never classified at all.
pd.DataFrame([
    {"id": 1, "best_detection_url": "http://x/1.jpg",
     "determination_name": "Noctuidae", "determination_score": 0.94},
    {"id": 2, "best_detection_url": "http://x/2.jpg",
     "determination_name": "Not Lepidoptera", "determination_score": 0.88},
    {"id": 3, "best_detection_url": "http://x/3.jpg",
     "determination_name": "Geometridae", "determination_score": 0.41},
    {"id": 4, "best_detection_url": "http://x/4.jpg",
     "determination_name": None, "determination_score": None},
]).to_csv(SOURCE_CSV, index=False)

print("== source ==")
print(pd.read_csv(SOURCE_CSV).to_string(index=False))
print(f"\ndropping: {NON_ORGANISM_DETERMINATIONS}")

print("\n== ingested ==")
occurrences = cf.ingest_occurrences(
    PROJECT_PATH, SOURCE_CSV, id_col="id", image_url_col="best_detection_url",
    numeric_cols=["determination_score"], drop=NON_ORGANISM_DETERMINATIONS)
print(occurrences.to_string(index=False))

assert len(occurrences) == 3, occurrences
assert "Not Lepidoptera" not in set(occurrences["determination_name"])
print("\n  3 of 4 rows: the non-Lepidoptera detection isn't an occurrence.")
print("  Occurrence 4 survives -- unclassified is not the same as classified as")
print("  debris, so a missing value never matches a drop rule. A low score (3)")
print("  survives too: that's a judgement to make at export, not here.")

print("\n== the archived import ==")
imports_dir = os.path.join(PROJECT_PATH, "imports")
for name in sorted(os.listdir(imports_dir)):
    rows = pd.read_csv(os.path.join(imports_dir, name))
    print(f"  {name}: {len(rows)} rows")
    assert len(rows) == 4, "the import must keep what the source sent"
print("  <- 4 rows. Nothing was lost; the source file is archived before it is")
print("     parsed, so a dropped row is always recoverable and auditable.")

print("\n== keeping everything (drop=None) ==")
everything = cf.ingest_occurrences(
    PROJECT_PATH, SOURCE_CSV, id_col="id", image_url_col="best_detection_url",
    numeric_cols=["determination_score"], drop=None)
print(f"  {len(everything)} occurrences  <- what you'd ingest to audit the "
      "classifier itself")

print("\n== a rule naming a column that isn't there ==")
try:
    cf.ingest_occurrences(PROJECT_PATH, SOURCE_CSV, id_col="id",
                          drop={"verdict": ["junk"]})
except KeyError as exc:
    print(f"  raises: {exc}")
    print("  <- loud on purpose. Silently matching nothing would read as 'there")
    print("     was none of that in here', which is the wrong thing to believe.")

print(f"\nproject left at {PROJECT_PATH}")
