"""
Long-table storage for metric values, current_rows, latest_values.

A metric is any derived value associated with an occurrence and a part: a
trait, a QC value, a human label, an embedding, a cluster assignment, an
outlier score.

Values are stored long, one row per occurrence-part-metric, because that is
what an interruptible run can append to, and reshaped wide on the way out.
Nothing is ever rewritten or deleted: a value measured from a mask that has
since been replaced stops being current but stays in the table, which is what
makes provenance answerable.
"""

import logging

import pandas as pd

from ..recipes import DEFAULT_PART, canonical_json, load_json
from . import masks as mask_records
from .runs import open_database

logger = logging.getLogger(__name__)


def make_metric_row(occurrence_id, part, metric_name, value, unit=None,
                    source_mask_hash=None):
    """
    Build one metric value record.

    occurrence_id    -- occurrence the value belongs to.
    part             -- part the value was measured on.
    metric_name      -- what it is stored under.
    value            -- the value. Usually a scalar; a dict reports several
                        related numbers and export splits it into one column per
                        key. Must be JSON-serializable.
    unit             -- what the value is expressed in, e.g. "px", "category".
    source_mask_hash -- identity of the MASK this was measured from, from
                        records.masks.derivation_hash(). Recorded per row rather
                        than on the run, because one run legitimately spans
                        occurrences whose masks came from different recipes.
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
    """
    Append metric values produced by one run.

    Written per occurrence as the run progresses rather than buffered to the
    end, so an interrupted run keeps everything it had already computed -- and
    so its next attempt correctly skips that work instead of redoing it.

    project_path -- project to write into.
    run_id       -- the run these values came from (see records.runs).
    recipe_hash  -- the recipe's hash, denormalized onto every row so the
                    repeat-check query never has to join back to runs.
    rows         -- list of records from make_metric_row().
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


def load_metrics(project_path, run_names=None, parts=None, metric_names=None):
    """
    Read metric values in LONG form -- one row per occurrence-part-metric --
    joined to the run that produced each.

    Long form is the honest shape of the underlying data and the right one for
    inspecting provenance (which run, which mask, which unit, when). Use
    export.metrics_wide() when you want one row per occurrence to analyze.

    The run's name, kind, and start time come along as run_name/run_kind/
    run_created_at, which is why no row stores its own copy of any of them.

    run_names    -- optional list of run names to include; all runs if None.
    parts        -- optional list of parts to include; all parts if None.
    metric_names -- optional list of metric names to include; all if None.
    """
    query = """
        SELECT m.*, r.name AS run_name, r.kind AS run_kind,
               r.created_at AS run_created_at
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
    """
    conditions = []
    parameters = []
    for column, values in (("r.name", run_names), ("m.part", parts),
                           ("m.metric_name", metric_names)):
        if values is not None:
            placeholders = ",".join("?" for _ in values)
            conditions.append(f"{column} IN ({placeholders})")
            parameters.extend(values)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with open_database(project_path) as connection:
        rows = [dict(row) for row in connection.execute(query, parameters)]

    for row in rows:
        row["value"] = load_json(row.pop("value_json"))

    if not rows:
        return pd.DataFrame(columns=[
            "metric_id", "run_id", "occurrence_id", "part", "metric_name",
            "value", "unit", "recipe_hash", "source_mask_hash", "run_name",
            "run_kind", "run_created_at",
        ])

    return pd.DataFrame(rows)


def current_rows(project_path, long_df):
    """
    Drop values measured from a mask the project has since replaced, leaving
    those that still describe what it holds.

    Long-form in, long-form out, so this composes onto any query.

    Canonical and reference masks are pooled into one currency test, because the
    long table doesn't record which table a run measured -- so a reference-mask
    value must not be discarded for failing to match a canonical mask it was
    never derived from.

    Two things are kept rather than judged: rows with no source_mask_hash, which
    are of unrecorded provenance rather than known-stale, and everything when the
    project has no mask table at all, since "no masks" must not mean "no values".
    """
    if long_df.empty:
        return long_df

    current = {}
    for reference in (False, True):
        hashes = mask_records.current_derivation_hashes(project_path,
                                                        reference=reference)
        for key, recipe_hash in hashes.items():
            current.setdefault(key, set()).add(recipe_hash)
    if not current:
        return long_df

    def is_current(row):
        # Anything non-str (None from sqlite, NaN if pandas widened the column)
        # is an unrecorded source, which is unjudgeable rather than stale.
        if not isinstance(row.source_mask_hash, str):
            return True
        return row.source_mask_hash in current.get(
            (row.occurrence_id, row.part), ())

    keep = pd.Series([is_current(row) for row in long_df.itertuples(index=False)],
                     index=long_df.index)
    superseded = int((~keep).sum())
    if superseded:
        logger.info("ignoring %d metric value(s) measured from a mask that has "
                    "since been replaced", superseded)
    return long_df[keep]


def latest_values(project_path, run_name, part=DEFAULT_PART, metric_name=None,
                  current_only=True):
    """
    The newest value per occurrence for one metric, as a Series indexed by
    occurrence_id -- the narrow lookup group metrics use to assemble a
    reference population's feature columns without reshaping the whole project
    wide first.

    Newest means last written: metric_id is the metrics table's insertion order,
    so it ranks two values computed within one run as well as two computed years
    apart, which a timestamp written once per batch could not.

    run_name     -- run that produced the values.
    part         -- part they were measured on.
    metric_name  -- metric to pull; required.
    current_only -- as in export.metrics_wide, and on by default for the same
                    reason with more at stake: a group metric fits a reference
                    population from these values, so a stale one doesn't just
                    misreport its own occurrence, it shifts the distribution
                    every other occurrence is scored against.
    """
    if metric_name is None:
        raise ValueError("latest_values needs a metric_name")

    long_df = load_metrics(project_path, run_names=[run_name], parts=[part],
                           metric_names=[metric_name])
    if current_only:
        long_df = current_rows(project_path, long_df)
    if long_df.empty:
        return pd.Series(dtype="object", name=metric_name)

    long_df = long_df.sort_values("metric_id")
    newest = long_df.drop_duplicates(subset=["occurrence_id"], keep="last")
    return newest.set_index("occurrence_id")["value"].rename(metric_name)
