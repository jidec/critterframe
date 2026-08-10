"""
Export: one row per occurrence, one column per trait, optionally filtered.

This is where a project stops being a processing pipeline and becomes a
dataset. Metric values are stored long and per-part; an export reshapes them
wide, joins whatever occurrence metadata you want alongside, and writes a CSV
an analysis can read.

The reshape itself lives here too (column_name, metrics_wide, metric_units),
because it's a view built for a reader rather than a way of storing anything.
Long per occurrence-part-metric is what the metric log IS -- see
records.metrics, which owns that -- and everything wide-form is a decision about
presenting it: how columns are named, which of several values for one cell wins,
and which values still describe the project's current masks. Validation and
dataset assembly build on the same view for the same reason.

Filtering happens HERE and only here. A filter is a rule for selecting
occurrences, not for deleting them: excluding blurred images from an export
leaves every blurred image, its mask, and its measurements exactly where they
were, so a threshold can be revised and the export rerun without recomputing
anything or losing anything. Two exports with two different filters can coexist
from one project, which is the point -- "which occurrences count" is an
analytical decision, and analytical decisions belong in the analysis rather than
baked into the data.

The one thing an export does decide for you is currency: it reports the values
measured from the masks the project currently holds, not any earlier value left
behind by a resegmentation (current_only, and see records.metrics). Which mask a
number came from isn't an analytical preference the way a threshold is -- a
value derived from a mask that no longer exists is describing something the
project doesn't contain.
"""

import logging
import operator

import pandas as pd

from .calibrations import scale as scale_calibration
from .project import paths, subsets as subset_selection
from .records.metrics import current_rows, load_metrics
from .records.occurrences import ID_COL

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
    independently and any two of them can collide: the same metric measured on
    head and thorax, or measured under two different recipes, must not land in
    the same column and silently overwrite each other. `key` is appended for
    dict-valued metrics, one column per key.
    """
    parts = [run_name, part, metric_name] + ([key] if key is not None else [])
    return "__".join(str(piece) for piece in parts)


def metrics_wide(project_path, run_names=None, parts=None, metric_names=None,
                 current_only=True):
    """
    Reshape stored metric values into one row per occurrence, one column per
    run/part/metric (see column_name) -- the shape an analysis wants, and what
    export and validation both build on.

    Here rather than in records.metrics because the reshape is an OUTPUT
    decision, not a storage one: long is what the metric log is, and wide is a
    view of it built for whoever is about to read it. Column naming, "newest
    wins", and dropping stale values are all part of that same view.

    Where a metric has been computed more than once for the same
    occurrence-part under the same run name -- a deliberate force=True rerun,
    or the same recipe rerun after a parameter change -- the NEWEST value wins,
    newest meaning last written (metric_id is the metric log's insertion order).
    Nothing is deleted; the older rows stay in the long table with their own
    run ids and recipe hashes, which is where to look when two numbers disagree
    and you need to know which recipe produced which.

    current_only -- report only values measured from the masks the project
                    currently holds (see records.metrics.current_rows). On by
                    default, and the reason is that "newest wins" alone is not
                    enough: a value from a superseded mask can be the newest one
                    there is, if the mask was replaced and the metric never
                    rerun. An analysis that picked it up would be describing a
                    mask the project no longer contains, with nothing in the
                    wide table to show for it. Pass False to see every value the
                    long table holds regardless of which mask produced it, which
                    is a provenance question -- and load_metrics answers those
                    better, since it keeps the mask hash beside the value.

    Returns a DataFrame with occurrence_id as its first column. Empty (just
    that column) if no values match.
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
    The unit recorded for each wide-form column -- so an export can carry its
    units somewhere rather than losing them in the reshape.

    Returns {column_name: unit}. A dict-valued metric's per-key columns all
    report the parent metric's unit, since the keys share it.

    Resolved exactly the way metrics_wide resolves values: current rows only,
    newest wins by metric_id. A column's unit has to describe the number that
    column actually holds, and it stopped being merely descriptive the moment
    unit-driven conversion existed -- reporting "px" for a value the export
    already divided into millimetres would be worse than reporting nothing.
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
    Convert the pixel columns of a wide export into millimetres, renaming each
    with the unit it now carries.

    df       -- wide table, one row per occurrence, indexed positionally with an
                occurrence_id column (what metrics_wide returns).
    unit_map -- {column: unit} from metric_units().
    scale    -- px_per_mm per occurrence, a Series indexed by occurrence_id
                (records.scales.scale_for_occurrences).

    `traits__organism__body_length` becomes `traits__organism__body_length_mm`,
    an area becomes `..._mm2`. The suffix is the point: a CSV can't carry units
    in its header any other way, and two exports of one project that differ only
    in units would otherwise be indistinguishable once the file is open in
    something else.

    An occurrence with no scale gets NaN in every converted column rather than
    an unconverted pixel value, which would be the same number meaning something
    entirely different in the same column.
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

    filters -- {column: (op, value)} or {column: predicate}. op is one of
               "<", "<=", ">", ">=", "==", "!=", "in", "not in" ("in"/"not in"
               take a container, e.g. ("in", ["usable"])); a predicate is a
               callable(series) -> boolean series, for anything the operator
               shorthand can't express.

    Filtering on a column that doesn't exist raises: a typo'd column name
    should be loud, not silently match nothing and hand back an empty export.

    A NaN in a filtered column never passes, whatever the operator. "This
    metric wasn't measured" must not quietly count as passing a != test -- an
    unmeasured occurrence is not a verified-good one.
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


def export_metrics(project_path, path=None, runs=None, parts=None,
                   metric_names=None, filters=None, occurrence_columns=None,
                   subset=None, drop_empty=True, current_only=True,
                   units=None):
    """
    Build the wide, one-row-per-occurrence trait table, optionally write it to
    CSV, and return it.

    project_path      -- project to export from.
    path              -- CSV to write. None returns the DataFrame without
                        writing, which is what you want when the next step is
                        more Python rather than a file for R.
    runs              -- run names to include; every run if None.
    parts             -- parts to include; every part if None.
    metric_names      -- metric names to include; all if None.
    filters           -- {column: (op, value)} applied after the table is fully
                        built (see apply_filters). Column names need their full
                        run/part prefix, the same as any other column here.
    occurrence_columns -- occurrence-table columns to join alongside the traits
                        (species, date, collection, image_url...). None joins
                        none, keeping the export to derived values only.
    subset            -- restrict to a named subset's occurrences.
    drop_empty        -- drop occurrences with no metric values at all. On by
                        default: a project mid-processing shouldn't export
                        thousands of empty rows. Turn it off when you need one
                        row per occurrence regardless, to see what's missing.
    current_only      -- export only values measured from the masks the project
                        currently holds. On by default: an export is a dataset,
                        and a number derived from a mask that was replaced and
                        never remeasured would sit in it indistinguishable from
                        one that describes the current mask. Those values aren't
                        gone -- nothing here deletes anything -- they're in the
                        long table with the mask hash that produced them
                        (records.metrics.load_metrics), which is where a
                        provenance question belongs. Pass False to export
                        whatever is newest regardless of which mask it came
                        from, e.g. to reproduce a table published before a
                        resegmentation.
    units             -- None (default) exports the stored pixel values
                        untouched. "mm" converts every px and px2 column into
                        millimetres using the project's scale table, renaming
                        each with an _mm or _mm2 suffix and carrying px_per_mm
                        alongside so the conversion is auditable (see
                        to_millimetres and records.calibrations).

                        The conversion lives here, at the last possible moment,
                        rather than in the stored values. A metric's unit is
                        part of its recipe hash, so measuring in millimetres
                        would make a re-calibration a different recipe and
                        invalidate every trait in the project; converting at
                        export means a corrected scale costs one re-export and
                        nothing else. Occurrences with no scale get NaN in the
                        converted columns, never an unconverted pixel value
                        sitting in a column labelled mm.

    Column names are "{run}__{part}__{metric}", with a fourth "__{key}"
    component for metrics returning several values at once. All three
    components are always present because all three vary independently -- the
    same metric measured on head and thorax, or under two differently
    configured runs, must not collide into one column.
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
