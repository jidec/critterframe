"""Stored metric values written straight to a project, for tests that read them back."""

from critterframe.metrics.dimensions import body_length
from critterframe.core.recipes import Recipe
from critterframe.records.metrics import append_metrics, make_metric_row
from critterframe.records.runs import start_run


def store_values(
    project_path, values, run_name="traits", part="organism", metric_name="body_length", unit="px"
):
    """Append {occurrence_id: value} under a fresh run of `run_name`."""
    recipe = Recipe("metric", run_name, [body_length()], part=part)
    run_id = start_run(project_path, recipe)
    append_metrics(
        project_path,
        run_id,
        recipe.hash,
        [
            make_metric_row(occurrence_id, part, metric_name, value, unit=unit)
            for occurrence_id, value in values.items()
        ],
    )
