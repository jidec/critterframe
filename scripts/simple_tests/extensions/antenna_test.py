"""
Antenna API connection and export round trip.

Checks credentials work and the export endpoint responds before a pipeline run
depends on it. Requesting an export costs server time, so the login check runs
first and the export is opt-in below.

Needs ANTENNA_EMAIL, ANTENNA_PW, and ANTENNA_PROJECT_ID in the environment (a
.env file at the repo root works).

Run from the repo root:
    python scripts/simple_tests/extensions/antenna_test.py
"""

import logging
import os
import tempfile

from critterframe.extensions.antenna_lighttraps import api

logging.basicConfig(level=logging.INFO)

REQUEST_EXPORT = False   # set True to actually request and download one

print("base URL:", api.BASE)
print("project id:", api.project_id())

session = api.get_session()
print("authenticated:", "Authorization" in session.headers)

response = session.get(f"{api.BASE}/exports/", timeout=30)
response.raise_for_status()
existing = response.json().get("results", [])
print(f"\n{len(existing)} existing export(s) visible:")
for export in existing[:5]:
    print(f"  id={export.get('id')} format={export.get('format')} "
          f"records={export.get('record_count')} "
          f"ready={bool(export.get('file_url'))}")

if REQUEST_EXPORT:
    destination = os.path.join(tempfile.gettempdir(), "antenna_export.csv")
    path = api.fetch_export(session, destination)
    print(f"\ndownloaded -> {path} ({os.path.getsize(path)} bytes)")
    print("ingest it with "
          "extensions.antenna_lighttraps.ingest.ingest_occurrences("
          "project_path, import_csv_path=path)")
else:
    print("\nREQUEST_EXPORT is False -- set it True to request a fresh export")
