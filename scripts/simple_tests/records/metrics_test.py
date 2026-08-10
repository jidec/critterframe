"""
Inspect a project's run and metric log.

Every stored value carries the run that produced it, the recipe hash behind that
run, and the hash of the mask it was measured from. This prints all three --
which is where to look when two numbers for one occurrence disagree, or when a
run you expected to skip its work went ahead and redid it.

Needs a project with a metric run behind it.

Run from the repo root:
    python scripts/simple_tests/records/metrics_test.py
"""

import logging

from critterframe.export import metrics_wide
from critterframe.records.metrics import load_metrics
from critterframe.records.runs import load_runs

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"

runs = load_runs(PROJECT_PATH)
if runs.empty:
    print("no runs recorded yet")
else:
    print("== runs, newest first ==")
    print(runs[["run_id", "kind", "name", "part", "recipe_hash", "status",
                "n_processed", "n_skipped", "n_failed"]].to_string(index=False))

    print("\n== the newest run's recipe, as stored ==")
    for index, operation in enumerate(runs.iloc[0]["recipe"]["operations"]):
        print(f"  {index}. {operation['kind']:<9} {operation['name']:<24} "
              f"v{operation['version']}  {operation['parameters']}")

values = load_metrics(PROJECT_PATH)
if values.empty:
    print("\nno metric values yet")
else:
    print(f"\n== {len(values)} stored value(s), long form ==")
    print(values[["occurrence_id", "part", "run_name", "metric_name", "value",
                  "unit", "recipe_hash", "source_mask_hash"]]
          .head(15).to_string(index=False))

    print("\n== the same values, reshaped wide ==")
    print(metrics_wide(PROJECT_PATH).head(10).to_string(index=False))
