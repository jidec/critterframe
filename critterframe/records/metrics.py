"""Metric records: long storage of values, and which of them are still current."""

import logging

import pandas as pd

from ..core.recipes import DEFAULT_PART, canonical_json, load_json
from . import masks as mask_records
from . import runs as run_records
from .runs import open_database

logger = logging.getLogger(__name__)

# The unit of a metric run's recorded transform info: one row per transform,
# named by its label, its scalar info as the value. Export leaves these columns
# out unless asked, and they are always filterable.
TRANSFORM_INFO_UNIT = "transform_info"

# Occurrence ids bound into one query by load_metrics.
OCCURRENCE_ID_CHUNK = 10_000


def make_metric_row(occurrence_id, part, metric_name, value, unit=None, source_mask_hash=None):
    """Build one metric value record.

    Args:
        occurrence_id: Occurrence the value belongs to.
        part: Part it was measured on.
        metric_name: Name it is stored under.
        value: The value: a JSON-serializable scalar, list or dict.
        unit: What the value is expressed in, e.g. `"px"`.
        source_mask_hash: Derivation hash of the mask it was measured from.
    """
    return {
        "occurrence_id": str(occurrence_id),
        "part": part,
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "source_mask_hash": source_mask_hash,
    }


def append_metrics(project_path, run_id, recipe_hash, rows):
    """Append the metric values one run produced.

    Args:
        project_path: Project to write into.
        run_id: The run the values came from.
        recipe_hash: The run's recipe hash, stored on every row.
        rows: Records from `make_metric_row`.
    """
    if not rows:
        return 0

    with open_database(project_path) as connection:
        connection.executemany(
            """
            INSERT INTO metrics (
                run_id, occurrence_id, part, metric_name, value_json,
                unit, recipe_hash, source_mask_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["occurrence_id"],
                    row["part"],
                    row["metric_name"],
                    canonical_json(row["value"]),
                    row["unit"],
                    recipe_hash,
                    row["source_mask_hash"],
                )
                for row in rows
            ],
        )

    return len(rows)


def load_metrics(
    project_path, run_names=None, parts=None, metric_names=None, occurrence_ids=None, current_only=False
):
    """Read metric values in long form, one row per occurrence-part-metric.

    Each row carries its run's name, kind and start time as `run_name`, `run_kind` and
    `run_created_at`.

    Args:
        project_path: Project to read from.
        run_names: Run names to include; all if None.
        parts: Parts to include; all if None.
        metric_names: Metric names to include; all if None.
        occurrence_ids: Occurrences to include; all if None.
        current_only: Keep only rows that are still current (see `current_rows`).
    """
    query = """
        SELECT m.*, r.name AS run_name, r.kind AS run_kind,
               r.created_at AS run_created_at
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
    """
    conditions = []
    parameters = []
    for column, values in (
        ("r.name", run_names),
        ("m.part", parts),
        ("m.metric_name", metric_names),
    ):
        if values is not None:
            placeholders = ",".join("?" for _ in values)
            conditions.append(f"{column} IN ({placeholders})")
            parameters.extend(values)

    # Occurrence ids are bound a chunk at a time: one parameter per occurrence
    # passes SQLite's variable limit on a large project.
    id_chunks = [None]
    if occurrence_ids is not None:
        unique_ids = list(dict.fromkeys(occurrence_ids))
        id_chunks = [
            unique_ids[start : start + OCCURRENCE_ID_CHUNK]
            for start in range(0, len(unique_ids), OCCURRENCE_ID_CHUNK)
        ]

    rows = []
    if run_records.has_database(project_path):
        with open_database(project_path) as connection:
            for id_chunk in id_chunks:
                chunk_conditions = list(conditions)
                chunk_parameters = list(parameters)
                if id_chunk is not None:
                    placeholders = ",".join("?" for _ in id_chunk)
                    chunk_conditions.append(f"m.occurrence_id IN ({placeholders})")
                    chunk_parameters.extend(id_chunk)
                chunk_query = query
                if chunk_conditions:
                    chunk_query += " WHERE " + " AND ".join(chunk_conditions)
                rows.extend(dict(row) for row in connection.execute(chunk_query, chunk_parameters))

    for row in rows:
        row["value"] = load_json(row.pop("value_json"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric_id",
                "run_id",
                "occurrence_id",
                "part",
                "metric_name",
                "value",
                "unit",
                "recipe_hash",
                "source_mask_hash",
                "run_name",
                "run_kind",
                "run_created_at",
            ]
        )

    long_df = pd.DataFrame(rows)
    return current_rows(project_path, long_df) if current_only else long_df


def current_rows(project_path, long_df):
    """Drop the values that no longer describe what the project holds.

    A row is current when its mask is still the project's and its recipe is the one its
    run name points at. Kept without judgement: a row with no `source_mask_hash`,
    everything in a project with no mask table, and a run name with no pointer.

    Args:
        project_path: Project to judge against.
        long_df: Long-form rows from `load_metrics`.

    Returns:
        The current rows, in the same form.
    """
    if long_df.empty:
        return long_df

    current_masks = {}
    for reference in (False, True):
        hashes = mask_records.current_derivation_hashes(project_path, reference=reference)
        for key, recipe_hash in hashes.items():
            current_masks.setdefault(key, set()).add(recipe_hash)

    pointers = run_records.current_recipe_pointers(project_path)

    def mask_is_current(row):
        if not current_masks:
            return True
        # Anything non-str (None from sqlite, NaN if pandas widened the column)
        # is an unrecorded source, which is unjudgeable rather than stale.
        if not isinstance(row.source_mask_hash, str):
            return True
        return row.source_mask_hash in current_masks.get((row.occurrence_id, row.part), ())

    def recipe_is_current(row):
        pointed = pointers.get((row.run_name, row.part))
        return pointed is None or pointed == row.recipe_hash

    keep = pd.Series(
        [mask_is_current(row) and recipe_is_current(row) for row in long_df.itertuples(index=False)],
        index=long_df.index,
    )
    superseded = int((~keep).sum())
    if superseded:
        logger.info(
            "ignoring %d metric value(s) either measured from a mask "
            "that has since been replaced or produced by a recipe "
            "their run_name has since moved off of",
            superseded,
        )
    return long_df[keep]


def result_keys(project_path, run_name, part, current_only=True):
    """Return which occurrences a metric run has a result for, without reading any value.

    Args:
        project_path: Project to read from.
        run_name: Run that produced the values.
        part: Part they were measured on.
        current_only: Only results that are still current.

    Returns:
        A DataFrame of `occurrence_id`, `part`, `recipe_hash`, `source_mask_hash` and
        `run_name`, one row per distinct combination.
    """
    columns = ["occurrence_id", "part", "recipe_hash", "source_mask_hash", "run_name"]
    if not run_records.has_database(project_path):
        return pd.DataFrame(columns=columns)
    with open_database(project_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
            SELECT DISTINCT m.occurrence_id, m.part, m.recipe_hash,
                   m.source_mask_hash, r.name AS run_name
            FROM metrics m
            JOIN runs r ON r.run_id = m.run_id
            WHERE r.name = ? AND m.part = ?
            """,
                (run_name, part),
            )
        ]
    keys = pd.DataFrame(rows, columns=columns)
    return current_rows(project_path, keys) if current_only else keys


# The most occurrence ids latest_values hands to the database as a filter. More
# than this and it reads the whole run and narrows afterwards: SQLite caps how
# many values one statement can bind.
MAX_FILTERED_IDS = 20000


def latest_values(
    project_path, run_name, part=DEFAULT_PART, metric_name=None, current_only=True, occurrence_ids=None
):
    """Return the newest value per occurrence for one metric.

    Args:
        project_path: Project to read from.
        run_name: Run that produced the values.
        part: Part they were measured on.
        metric_name: Metric to read. Required.
        current_only: Only values that are still current.
        occurrence_ids: Occurrences to read; all if None. Worth passing for a vector
            metric, where reading every stored value is slow.

    Returns:
        A Series indexed by occurrence id.
    """
    if metric_name is None:
        raise ValueError("latest_values needs a metric_name")

    wanted = None if occurrence_ids is None else {str(i) for i in occurrence_ids}
    in_query = wanted is not None and len(wanted) <= MAX_FILTERED_IDS
    long_df = load_metrics(
        project_path,
        run_names=[run_name],
        parts=[part],
        metric_names=[metric_name],
        occurrence_ids=sorted(wanted) if in_query else None,
    )
    if wanted is not None and not in_query and not long_df.empty:
        long_df = long_df[long_df["occurrence_id"].isin(wanted)]
    if current_only:
        long_df = current_rows(project_path, long_df)
    if long_df.empty:
        return pd.Series(dtype="object", name=metric_name)

    long_df = long_df.sort_values("metric_id")
    newest = long_df.drop_duplicates(subset=["occurrence_id"], keep="last")
    return newest.set_index("occurrence_id")["value"].rename(metric_name)
