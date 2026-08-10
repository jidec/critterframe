"""
Metric values and their provenance.

A metric is any derived value associated with an occurrence and a part: a
trait, a quality-check value, a human label, an embedding, a cluster
assignment, an outlier score. Group metrics are metrics too -- being fit
against a reference population changes how a value is COMPUTED, not what it is
once computed -- so they store here like everything else.

Values are stored long (one row per occurrence-part-metric) and reshaped wide
on the way out, because long is what an interruptible run can append to and
wide is what an analysis wants. The value itself is JSON, so a metric returning
a dict, a list of embedding floats, or a bare number all store the same way
without a schema change.

Every row carries the recipe hash that produced it AND the recipe hash of the
mask it was measured from, which together are what makes expensive metrics
behave like cached derived data rather than disposable calculations: a rerun of
the same recipe over the same masks finds its own rows already present and skips
them, while a rerun over a mask that has since been replaced does not.

Values are never rewritten or deleted. A metric is an immutable historical
result: replacing an occurrence's canonical mask leaves every number derived
from the old one exactly where it is, it just stops being CURRENT for that
occurrence-part. So the long table accumulates

    123 | black_fraction | .31 | recipe X | mask A     <- superseded, kept
    123 | black_fraction | .27 | recipe X | mask B     <- current

while masks.parquet holds only mask B. That keeps the provenance of a number
that was already published, and still lets the working analysis follow the
current masks: everything that reshapes values for analysis (export's
metrics_wide, latest_values here) reports only the rows whose source mask is the
one the project currently holds.

A row stores only what isn't already recorded once per run. There's no per-row
version -- a metric operation's version is in the recipe spec, so it's in the
recipe hash on every row and in the recipe JSON on the run -- and no per-row
timestamp, since the run has one and metric_id already carries insertion order.
Storing either again would be a second copy of the same provenance, free to
drift and not obviously authoritative when it did.
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
    metric_name      -- what it's stored under (a Metric operation's
                        metric_name, which defaults to its operation name).
    value            -- the value itself. Usually a scalar; a dict is allowed
                        and is how one metric reports several related numbers,
                        which export splits into one column per key. Must be
                        JSON-serializable.
    unit             -- what the value is expressed in ("px", "px2",
                        "category", ...).
    source_mask_hash -- identity of the MASK this value was measured from:
                        records.masks.derivation_hash(), which is the mask's
                        segmentation recipe hash for a mask found straight in
                        the image and a chained hash for one derived from
                        another part's mask.

                        Recorded per row rather than only on the run because a
                        single metric run legitimately spans occurrences whose
                        masks came from different segmentation recipes (two
                        subsets processed differently, or a resegmented
                        subset), so "which mask produced this number" is a
                        property of the row, not of the run.
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
    Drop the values that were measured from a mask the project has since
    replaced, leaving the ones that still describe what it currently holds.

    Long-form in, long-form out (see load_metrics), so this composes onto any
    query rather than being welded into one reshape.

    Canonical and reference masks are pooled into one "is this hash current for
    this occurrence-part" test, because the long table doesn't record which of
    the two tables a run measured -- inputs={"masks": ...} lives in the run's
    recipe, not on the value -- and a reference-mask value must not be thrown
    away for failing to match a canonical mask it was never derived from. What
    pooling costs is that a value would survive on a match against the other
    table's mask -- which takes one recipe hash writing into both tables, and
    can't happen through run_segments, since inputs={"masks": ...} puts the
    table into the hash. It errs toward keeping a real value either way.

    Two things are deliberately kept rather than judged:

      - rows with no source_mask_hash, which are of unrecorded provenance rather
        than known-stale;
      - everything, when the project has no mask table at all, since there is
        nothing to compare against and "no masks" must not mean "no values".
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
