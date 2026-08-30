"""
Export a one-row-per-occurrence trait table, optionally filtered; select
occurrences by stored values.

The long-to-wide reshape lives here too (column_name, metrics_wide,
metric_units), because it is a view built for a reader rather than a way of
storing anything -- validation and dataset assembly build on the same view.

Filtering happens here and only here. A filter selects occurrences, it never
deletes them, so a threshold can be revised and the export rerun without
recomputing anything. Exports report values measured from the masks the project
currently holds (current_only), since a value from a mask that no longer exists
describes something the project doesn't contain.
"""

import logging
import operator

import pandas as pd

from .calibrations import scale as scale_calibration
from .project import paths, subsets as subset_selection
from .recipes import DEFAULT_PART
from .records.metrics import current_rows, load_metrics
from .records.occurrences import ID_COL
from .selectionhelpers import rows_matching

logger = logging.getLogger(__name__)

# Operators available to the `filters` dict, keyed by the strings you'd write by
# hand. "in"/"not in" are handled separately since they take a container rather
# than a scalar on the right-hand side.
COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def column_name(run_name, part, metric_name, key=None):
    """
    The wide-form column one metric value lands in.

    Run name, part, and metric name are all included because all three vary
    independently and any two can collide -- the same metric on head and thorax,
    or under two recipes, must not overwrite each other. `key` is appended for
    dict-valued metrics, one column per key.
    """
    parts = [run_name, part, metric_name] + ([key] if key is not None else [])
    return "__".join(str(piece) for piece in parts)


def metrics_wide(project_path, run_names=None, parts=None, metric_names=None,
                 current_only=True):
    """
    Reshape stored metric values into one row per occurrence, one column per
    run/part/metric.

    Here rather than in records.metrics because the reshape is an output
    decision, not a storage one. Where a metric was computed more than once for
    the same occurrence-part under one run name, the NEWEST value wins.

    current_only -- report only values measured from the masks the project
                    currently holds. On by default: "newest wins" alone isn't
                    enough, since a value from a superseded mask can be the
                    newest there is if the metric was never rerun. False shows
                    every value regardless, which is a provenance question that
                    load_metrics answers better.

    Returns a DataFrame with occurrence_id first; empty if nothing matches.
    """
    long_df = load_metrics(project_path, run_names=run_names, parts=parts,
                           metric_names=metric_names)
    if current_only:
        long_df = current_rows(project_path, long_df)
    if long_df.empty:
        return pd.DataFrame(columns=[ID_COL])

    long_df = long_df.sort_values("metric_id")

    wide = {}
    for row in long_df.itertuples(index=False):
        columns = wide.setdefault(row.occurrence_id, {})
        if isinstance(row.value, dict):
            for key, subvalue in row.value.items():
                columns[column_name(row.run_name, row.part, row.metric_name, key)] = subvalue
        else:
            columns[column_name(row.run_name, row.part, row.metric_name)] = row.value

    wide_df = pd.DataFrame.from_dict(wide, orient="index")
    wide_df.index.name = ID_COL
    wide_df = wide_df.reset_index()
    return wide_df


def metric_units(project_path, run_names=None, current_only=True):
    """
    The unit recorded for each wide-form column, as {column_name: unit}, so an
    export can carry its units.

    Resolved exactly as metrics_wide resolves values: current rows only, newest
    wins. A column's unit has to describe the number that column holds --
    reporting "px" for a value already converted to millimetres would be worse
    than reporting nothing.
    """
    long_df = load_metrics(project_path, run_names=run_names)
    if current_only:
        long_df = current_rows(project_path, long_df)
    if long_df.empty:
        return {}

    units = {}
    for row in long_df.sort_values("metric_id").itertuples(index=False):
        keys = row.value.keys() if isinstance(row.value, dict) else [None]
        for key in keys:
            units[column_name(row.run_name, row.part, row.metric_name, key)] = row.unit
    return units


# Pixel units and what they become once a px/mm scale is applied: the suffix a
# converted column takes, and the power of the scale it's divided by. A length
# divides once, an area twice. Anything not listed -- a fraction, a category, an
# embedding, a laplacian variance -- has no physical length in it to convert and
# is left exactly as it is.
CONVERTIBLE_UNITS = {"px": ("mm", 1), "px2": ("mm2", 2)}


def to_millimetres(df, unit_map, scale):
    """
    Convert pixel columns to millimetres using each occurrence's px/mm scale.

    Lengths divide by the scale once and areas twice; anything with no physical
    length in it (a fraction, a category, an embedding) is left alone.
    Converted columns are renamed with an _mm/_mm2 suffix and the divisor is
    carried alongside.

    An occurrence with no calibration gets NaN rather than an unconverted pixel
    value sitting in a column labelled mm.
    """
    scale = scale.reindex(df[ID_COL].astype(str)).to_numpy(dtype="float64")

    converted = {}
    untouched = []
    for column in df.columns:
        if column == ID_COL:
            continue
        conversion = CONVERTIBLE_UNITS.get(unit_map.get(column))
        if conversion is None:
            if column in unit_map:
                untouched.append(column)
            continue
        suffix, power = conversion
        converted[column] = (f"{column}_{suffix}",
                             pd.to_numeric(df[column], errors="coerce")
                             / (scale ** power))

    if not converted:
        logger.warning("units='mm' but no column is in px or px2 -- nothing to "
                       "convert (units seen: %s)",
                       sorted({unit_map.get(c) for c in df.columns
                               if c in unit_map}))
        return df

    out = df.copy()
    for column, (renamed, values) in converted.items():
        out[renamed] = values
        out = out.drop(columns=[column])

    # The divisor rides along: a millimetre in the table is only as good as the
    # calibration behind it, and someone reading the CSV a year later needs to
    # be able to see which one was used without going back to the project.
    out[scale_calibration.SCALE_COL] = scale

    missing = int(pd.isna(scale).sum())
    if missing:
        logger.warning("%d of %d occurrence(s) have no scale, so their %d "
                       "converted column(s) are NaN", missing, len(out),
                       len(converted))
    if untouched:
        logger.info("left %d non-length column(s) as they were: %s",
                    len(untouched), ", ".join(sorted(untouched)[:5])
                    + ("..." if len(untouched) > 5 else ""))

    return out


def apply_filters(df, filters):
    """
    Narrow df to the rows passing every condition, ANDed together.

    filters -- {column: (op, value)} or {column: predicate}. op is one of "<",
               "<=", ">", ">=", "==", "!=", "in", "not in"; a predicate is a
               callable(series) -> boolean series.

    Filtering on a column that doesn't exist raises rather than silently
    matching nothing. A NaN never passes, whatever the operator -- "this metric
    wasn't measured" must not count as passing a != test.
    """
    keep = pd.Series(True, index=df.index)

    for column, condition in filters.items():
        if column not in df.columns:
            raise KeyError(
                f"filter column '{column}' not in the export "
                f"(available: {sorted(df.columns)})"
            )
        series = df[column]

        if callable(condition):
            passes = condition(series)
        else:
            op, value = condition
            if op == "in":
                passes = series.isin(value)
            elif op == "not in":
                passes = ~series.isin(value)
            elif op in COMPARATORS:
                passes = COMPARATORS[op](series, value)
            else:
                raise ValueError(
                    f"unsupported filter op {op!r} for column {column!r} -- "
                    f"expected a callable or one of "
                    f"{sorted(COMPARATORS) + ['in', 'not in']}"
                )

        keep &= passes.fillna(False) & series.notna()

    logger.info("filters kept %d of %d occurrences", int(keep.sum()), len(df))
    return df[keep]


def occurrences_matching(project_path, run_name, rules, part=DEFAULT_PART,
                         current_only=False):
    """
    The occurrences whose stored metric values match a {metric: values} rule set
    -- "the ones a person called usable", "the ones a classifier called debris".

    Rules take BARE metric names; the run and part prefixes are added for you.
    Any rule matching is enough, a missing value never matches, and a metric
    name that produced no column raises rather than selecting nothing. A run
    with no stored values at all is the one empty case that isn't a typo, and
    returns [] with a warning.

    project_path -- project to read from.
    run_name     -- run whose values the rules are written against.
    rules        -- {metric_name: value} or {metric_name: [values...]}.
    part         -- part the values were recorded for.
    current_only -- False by default, against the grain of everything else that
                    reads stored values: a label like "cut_off" describes the
                    CROP and stays true whatever mask was on screen. Left at
                    True, resegmenting would void a whole review session. Pass
                    True where a rule reads a value that describes a mask.

    Returns a sorted list of occurrence ids.
    """
    rules = {column_name(run_name, part, metric_name): values
             for metric_name, values in rules.items()}

    df = metrics_wide(project_path, run_names=[run_name], parts=[part],
                      current_only=current_only)
    if df.empty:
        logger.warning("run '%s' has no stored values for part '%s' -- nothing "
                       "to match %s against, selecting none", run_name, part,
                       sorted(rules))
        return []

    matched = df[rows_matching(df, rules)]
    logger.info("%d of %d occurrence(s) in run '%s' match %s",
                len(matched), len(df), run_name, rules)
    return sorted(matched[ID_COL].astype(str))


def export_metrics(project_path, path=None, runs=None, parts=None,
                   metric_names=None, filters=None, occurrence_columns=None,
                   subset=None, drop_empty=True, current_only=True,
                   units=None):
    """
    Build the wide, one-row-per-occurrence trait table, optionally write it to
    CSV, and return it.

    Order of operations matters: metadata is joined, empty columns dropped,
    units converted, then filters applied -- so a threshold written in
    millimetres filters millimetres.

    project_path      -- project to export from.
    path              -- CSV to write. None returns the DataFrame unwritten.
    runs              -- run names to include; every run if None.
    parts             -- parts to include; every part if None.
    metric_names      -- metric names to include; all if None.
    filters           -- {column: (op, value)}; see apply_filters.
    occurrence_columns-- occurrence-table columns to join alongside the traits.
    subset            -- restrict to a named subset.
    drop_empty        -- drop columns that came out entirely empty.
    current_only      -- only values measured from the project's current masks.
    units             -- "mm" converts pixel columns using each occurrence's
                         calibration; None leaves everything in pixels.

    Returns the exported DataFrame.
    """
    paths.require_project(project_path)

    df = metrics_wide(project_path, run_names=runs, parts=parts,
                      metric_names=metric_names, current_only=current_only)
    metric_columns = [column for column in df.columns if column != ID_COL]

    if subset is not None or occurrence_columns is not None or not drop_empty:
        occurrences = subset_selection.select_occurrences(
            project_path, subset=subset,
            columns=list(occurrence_columns) if occurrence_columns else [ID_COL],
        )
        # Left join, always: the subset restriction is already carried by the
        # left side, so `how` only decides whether an unmeasured occurrence
        # survives -- and that's drop_empty's decision alone. (An inner join
        # here would make drop_empty=False silently return nothing at all
        # before any metric has been run, which is exactly the moment someone
        # asks for every occurrence regardless.)
        df = occurrences.merge(df, on=ID_COL, how="left" if not drop_empty else "inner")

    if drop_empty and metric_columns:
        measured = df[metric_columns].notna().any(axis=1)
        dropped = int((~measured).sum())
        if dropped:
            logger.info("dropped %d occurrence(s) with no metric values", dropped)
        df = df[measured]

    # Converted after drop_empty so that "has any measurement" is judged on the
    # stored values: an occurrence measured perfectly well but lacking a
    # calibration should appear with empty millimetre columns, not vanish.
    # Before filters, so a threshold can be written against the mm column names.
    if units is not None:
        if units != "mm":
            raise ValueError(
                f"units={units!r} isn't supported -- 'mm' converts px/px2 "
                "columns, None (the default) exports stored pixel values"
            )
        scale = scale_calibration.scale_for_occurrences(project_path,
                                                    occurrence_ids=df[ID_COL])
        if scale.empty or not scale.notna().any():
            raise ValueError(
                "units='mm' but this project has no scale covering these "
                "occurrences, so every converted column would be empty. "
                "Record one with declare_scale() for a rig whose px/mm you "
                "know, or measure_scales() for images with a target in frame."
            )
        df = to_millimetres(
            df,
            metric_units(project_path, run_names=runs, current_only=current_only),
            scale,
        )
        metric_columns = [
            next((f"{column}_{suffix}" for suffix, _ in CONVERTIBLE_UNITS.values()
                  if f"{column}_{suffix}" in df.columns), column)
            for column in metric_columns
        ]

    if filters:
        df = apply_filters(df, filters)

    # occurrence_id first, then joined metadata, then the traits -- so the
    # identifying columns are on the left where anyone opening the CSV expects.
    ordered = [ID_COL] + [c for c in df.columns if c not in metric_columns
                          and c != ID_COL] + [c for c in metric_columns
                                              if c in df.columns]
    df = df[ordered].reset_index(drop=True)

    if path is not None:
        df.to_csv(path, index=False)
        logger.info("exported %d occurrences x %d columns -> %s",
                    len(df), len(df.columns), path)

    return df


def export_units(project_path, runs=None):
    """
    The unit behind each exported column, as {column: unit}.

    A CSV can't carry units in its header without mangling the column names, so
    they're available separately -- write them beside the export when handing
    the data to someone who wasn't there when it was measured. Every number in
    an export is in pixels, a fraction, or a category, and which one is not
    guessable from the column name alone.
    """
    return metric_units(project_path, run_names=runs)
