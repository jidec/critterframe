"""
Project summaries: what does this project actually contain right now.

Answers the questions you ask when you come back to a project after a month:
how many occurrences, how many have images, which parts have been segmented,
which runs have happened, how many metric values exist etc.

Every count here is read straight off the stored tables and not stored itself
"""

import logging

from ..records import masks as mask_records
from ..records.metrics import load_metrics
from ..records.occurrences import load_occurrences
from ..records.runs import load_runs
from ..storage.imagestore import ImageStore
from . import paths

logger = logging.getLogger(__name__)


def summarize(project_path):
    """
    A dict summarizing the project's contents.

    Deliberately returns data rather than printing, so it can back a status
    line, a test, or a report as easily as the console output print_summary()
    produces from it.
    """
    paths.require_project(project_path)

    occurrences = load_occurrences(project_path, missing_ok=True)

    n_images = 0
    if paths.images_path(project_path).exists():
        with ImageStore(project_path, readonly=True) as store:
            n_images = len(store.keys())

    summary = {
        "project_path": str(paths.project_dir(project_path)),
        "occurrences": len(occurrences),
        "images": n_images,
        "parts": {},
        "reference_parts": {},
        "runs": {},
        "metrics": {},
    }

    for reference, key in ((False, "parts"), (True, "reference_parts")):
        masks = mask_records.load_masks(project_path, reference=reference,
                                        columns=["occurrence_id", "part"])
        if not masks.empty:
            summary[key] = masks.groupby("part").size().to_dict()

    runs = load_runs(project_path)
    if not runs.empty:
        summary["runs"] = {
            "total": len(runs),
            "by_kind": runs.groupby("kind").size().to_dict(),
            "unfinished": int((runs["status"] != "complete").sum()),
            "latest": runs.iloc[0]["name"],
        }

    metrics = load_metrics(project_path)
    if not metrics.empty:
        summary["metrics"] = {
            "values": len(metrics),
            "names": sorted(metrics["metric_name"].unique()),
            "occurrences_measured": int(metrics["occurrence_id"].nunique()),
        }

    return summary


def print_summary(project_path):
    """Print summarize()'s result in a readable block. Returns the summary too."""
    summary = summarize(project_path)

    print(f"project: {summary['project_path']}")
    print(f"  occurrences   : {summary['occurrences']}")
    print(f"  images        : {summary['images']}")

    for label, key in (("masks", "parts"), ("reference masks", "reference_parts")):
        parts = summary[key]
        if parts:
            detail = ", ".join(f"{part}={count}" for part, count in sorted(parts.items()))
            print(f"  {label:<14}: {detail}")

    if summary["runs"]:
        runs = summary["runs"]
        kinds = ", ".join(f"{kind}={count}" for kind, count in sorted(runs["by_kind"].items()))
        print(f"  runs          : {runs['total']} ({kinds}), "
              f"{runs['unfinished']} unfinished, latest '{runs['latest']}'")

    if summary["metrics"]:
        metrics = summary["metrics"]
        print(f"  metric values : {metrics['values']} over "
              f"{metrics['occurrences_measured']} occurrences")
        print(f"  metric names  : {', '.join(metrics['names'])}")

    return summary
