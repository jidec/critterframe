"""
Summarize what a project directory currently holds.
"""

import logging

from ..recipes import DEFAULT_PART, describe_spec
from ..records import masks as mask_records
from ..records.metrics import TRANSFORM_INFO_UNIT, load_metrics
from ..records.occurrences import load_occurrences
from ..records.runs import current_recipe_pointers, load_runs
from ..storage.imagestore import ImageStore
from . import paths

logger = logging.getLogger(__name__)


def summarize(project_path, head=6):
    """
    A dict summarizing the project's contents.

    Deliberately returns data rather than printing, so it can back a status
    line, a test, or a report as easily as the console output print_summary()
    produces from it. occurrence_preview is the one exception: an R
    head()-style rendering of the occurrence table (row index, column header,
    aligned values, every column) is itself the useful "data" here -- there
    is no further reshaping a caller would do with the raw rows that a
    pre-rendered glance doesn't already serve, and pre-rendering keeps this
    dict JSON-friendly (a DataFrame isn't).

    - `head` -- rows to preview, R's head() default of 6.
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
        "occurrence_preview": (
            occurrences.head(head).to_string() if not occurrences.empty else ""),
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
            "by_name": _runs_by_name(project_path, runs),
        }

    metrics = load_metrics(project_path)
    if not metrics.empty:
        # A metric run's recorded transform info is kept apart: it describes
        # how a value was measured, not a value anyone asked for.
        is_info = metrics["unit"] == TRANSFORM_INFO_UNIT
        values = metrics[~is_info]
        summary["metrics"] = {
            "values": len(values),
            "names": sorted(values["metric_name"].unique()),
            "occurrences_measured": int(values["occurrence_id"].nunique()),
            "transform_info": sorted(metrics.loc[is_info, "metric_name"].unique()),
        }

    return summary


def _runs_by_name(project_path, runs):
    """
    One row per distinct (kind, name, part) in `runs` (already loaded,
    newest-first): how many times it's been run, its latest recipe, and --
    for a metric, where current_recipe_pointers has one -- whether the
    latest run is still the one a name currently means. A segment run_name
    cycling through several recipe hashes over a project's life isn't
    ambiguous the way a metric's is (see records.runs.resolve_recipe_currency),
    so its latest run always counts as current.

    Returns a list of dicts, not a dict keyed by (kind, name, part) -- summarize()
    promises a JSON-friendly result, and a tuple key isn't one.
    """
    pointers = current_recipe_pointers(project_path, kind="metric")
    rows = []
    for (kind, name, part), group in runs.groupby(["kind", "name", "part"], sort=False):
        latest = group.iloc[0]
        current_hash = (pointers.get((name, part)) if kind == "metric"
                        else latest["recipe_hash"])
        rows.append({
            "kind": kind,
            "name": name,
            "part": part,
            "n_runs": len(group),
            "latest_run_id": int(latest["run_id"]),
            "latest_recipe_hash": latest["recipe_hash"],
            "latest_created_at": latest["created_at"],
            "latest_status": latest["status"],
            "current_recipe_hash": current_hash,
            "description": describe_spec(latest["recipe"]),
        })
    rows.sort(key=lambda row: (row["kind"], row["name"], row["part"]))
    return rows


def print_summary(project_path, head=6):
    """
    Print summarize()'s result in a readable block. Returns the summary too.

    - `head` -- rows of the occurrence table to preview; see summarize().
    """
    summary = summarize(project_path, head=head)

    print(f"project: {summary['project_path']}")
    print(f"  occurrences   : {summary['occurrences']}")
    print(f"  images        : {summary['images']}")

    if summary["occurrence_preview"]:
        print(f"  occurrences (head of {min(head, summary['occurrences'])} "
              f"of {summary['occurrences']}):")
        for line in summary["occurrence_preview"].splitlines():
            print(f"    {line}")

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
        for row in runs["by_name"]:
            current = (" [current]" if row["current_recipe_hash"] == row["latest_recipe_hash"]
                      else f" [superseded -- current is {row['current_recipe_hash']}]")
            print(f"    {row['kind']:<7} '{row['name']}' ({row['part']}): "
                  f"{row['n_runs']} run(s), recipe {row['latest_recipe_hash']}{current}, "
                  f"latest {row['latest_created_at']}")

    if summary["metrics"]:
        metrics = summary["metrics"]
        print(f"  metric values : {metrics['values']} over "
              f"{metrics['occurrences_measured']} occurrences")
        print(f"  metric names  : {', '.join(metrics['names'])}")
        if metrics["transform_info"]:
            print(f"  transform info: {', '.join(metrics['transform_info'])}")

    return summary


def describe_run(project_path, run_id=None, name=None, part=DEFAULT_PART, kind=None):
    """
    Print one run's full recipe and context as readable text, and return the
    run as a dict.

    Everything printed here was already stored at start_run -- this only
    renders it: the operation chain a hash like "a1b2c3..." actually names,
    what occurrences it covered, and whether it's still the recipe its
    run_name currently means.

    - `project_path` -- project to read from.
    - `run_id` -- an exact run, from load_runs() or a "runs :" summary line.
      Takes priority over name/part/kind when given.
    - `name` -- run_name to resolve to its newest run, when run_id isn't given.
    - `part` -- part to resolve name against; the whole organism by default.
    - `kind` -- optional "segment"/"metric" filter, for a name shared by both.
    """
    if run_id is not None:
        runs = load_runs(project_path, run_id=run_id)
    else:
        if name is None:
            raise ValueError("describe_run needs run_id, or name (with part/kind)")
        runs = load_runs(project_path, kind=kind, name=name)
        runs = runs[runs["part"] == part]

    if runs.empty:
        raise KeyError(f"no run found for run_id={run_id!r} name={name!r} "
                       f"part={part!r} kind={kind!r}")

    row = runs.iloc[0].to_dict()

    current_hash = row["recipe_hash"]
    if row["kind"] == "metric":
        current_hash = current_recipe_pointers(
            project_path, kind="metric").get((row["name"], row["part"]))
    current = ("current" if current_hash == row["recipe_hash"]
              else f"superseded -- current is {current_hash}")

    print(f"run {row['run_id']}: {row['kind']} '{row['name']}' part={row['part']}"
          + (f" subset={row['subset']}" if row["subset"] else ""))
    print(f"  status        : {row['status']}  ({row['created_at']} "
          f"-> {row['finished_at']})")
    print(f"  processed/skipped/failed: {row['n_processed']}/{row['n_skipped']}/"
          f"{row['n_failed']}")
    print(f"  recipe_hash   : {row['recipe_hash']}  [{current}]")
    print(f"  pipeline      : {describe_spec(row['recipe'])}")
    for operation in row["recipe"]["operations"]:
        model = f"  model={operation['model']}" if "model" in operation else ""
        print(f"    - {operation['name']} v{operation['version']} "
              f"{operation['parameters']}{model}")
    if row["context"]:
        print(f"  context       : {row['context']}")

    return row
