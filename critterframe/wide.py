"""The wide view: current stored metric values reshaped to one row per occurrence, and filters over it."""

import logging
import operator
from typing import NamedTuple

import pandas as pd

from .calibrations import scale as scale_calibration
from .records import runs as run_records
from .records.metrics import TRANSFORM_INFO_UNIT, load_metrics
from .records.occurrences import ID_COL
from .selection import subsets as subset_selection

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
    """Return the wide-form column one metric value lands in: `<run>__<part>__<metric>[__<key>]`.

    Args:
        run_name: The run that measured it.
        part: The part measured.
        metric_name: The metric.
        key: One key of a dict-valued metric.
    """
    parts = [run_name, part, metric_name] + ([key] if key is not None else [])
    return "__".join(str(piece) for piece in parts)


def _current_long(project_path, run_names=None, parts=None, metric_names=None, current_only=True):
    """Return the stored metric values an export is built from, long, current ones only by default."""
    return load_metrics(
        project_path, run_names=run_names, parts=parts, metric_names=metric_names, current_only=current_only
    )


def _wide_from_long(long_df):
    """Reshape a long frame into one row per occurrence.

    Args:
        long_df: Rows from `_current_long`.

    Returns:
        A DataFrame with `occurrence_id` first; empty if there are no rows.
    """
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


def _column_provenance(long_df):
    """Return what each wide-form column holds, as `{column: {run_name, part, metric_name, key, unit}}`.

    Args:
        long_df: Rows from `_current_long`.
    """
    provenance = {}
    for row in long_df.sort_values("metric_id").itertuples(index=False):
        keys = row.value.keys() if isinstance(row.value, dict) else [None]
        for key in keys:
            provenance[column_name(row.run_name, row.part, row.metric_name, key)] = {
                "run_name": row.run_name,
                "part": row.part,
                "metric_name": row.metric_name,
                "key": key,
                "unit": row.unit,
            }
    return provenance


def metrics_wide(
    project_path, run_names=None, parts=None, metric_names=None, current_only=True, transform_info=False
):
    """Reshape stored metric values into one row per occurrence, one column per run, part and metric.

    Where a metric was stored more than once under one run name, the newest value wins.

    Args:
        project_path: Project to read from.
        run_names: Run names to include; all if None.
        parts: Parts to include; all if None.
        metric_names: Metric names to include; all if None.
        current_only: Only values measured from the masks the project currently holds.
        transform_info: Include what each metric run's transforms recorded.

    Returns:
        A DataFrame with `occurrence_id` first; empty if nothing matches.
    """
    long_df = _current_long(
        project_path, run_names=run_names, parts=parts, metric_names=metric_names, current_only=current_only
    )
    if not transform_info and metric_names is None and not long_df.empty:
        long_df = long_df[long_df["unit"] != TRANSFORM_INFO_UNIT]
    return _wide_from_long(long_df)


def metric_units(project_path, run_names=None, current_only=True):
    """Return the unit recorded for each wide-form column, as `{column: unit}`.

    Args:
        project_path: Project to read from.
        run_names: Run names to include; all if None.
        current_only: Only values measured from the masks the project currently holds.
    """
    provenance = _column_provenance(
        _current_long(project_path, run_names=run_names, current_only=current_only)
    )
    return {column: info["unit"] for column, info in provenance.items()}


def apply_filters(df, filters):
    """Narrow a frame to the rows passing every condition.

    Args:
        df: The frame.
        filters: As `passing` takes them.
    """
    keep = passing(df, filters)
    logger.info("filters kept %d of %d occurrences", int(keep.sum()), len(df))
    return df[keep]


def passing(df, filters):
    """Return which rows pass every condition, as a boolean Series.

    A missing value never passes, whatever the operator.

    Args:
        df: The frame.
        filters: `{column: (op, value)}` or `{column: predicate}`. `op` is one of `<`, `<=`,
            `>`, `>=`, `==`, `!=`, `in`, `not in`; a predicate takes a Series and returns a
            boolean Series.

    Raises:
        KeyError: If a filter names a column the frame doesn't have.
    """
    keep = pd.Series(True, index=df.index)

    for column, condition in filters.items():
        if column not in df.columns:
            raise KeyError(f"filter column '{column}' not in the export (available: {sorted(df.columns)})")
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

    return keep


# Pixel units and what they become once a px/mm scale is applied: the suffix a
# converted column takes, and the power of the scale it's divided by. A length
# divides once, an area twice. Anything not listed -- a fraction, a category, an
# embedding, a laplacian variance -- has no physical length in it to convert and
# is left exactly as it is.
CONVERTIBLE_UNITS = {"px": ("mm", 1), "px2": ("mm2", 2)}


def converted_name(column, columns):
    """Return the name a pixel column has after conversion to millimeters.

    Args:
        column: The column's name before conversion.
        columns: The columns of the converted frame.
    """
    for suffix, _power in CONVERTIBLE_UNITS.values():
        if f"{column}_{suffix}" in columns:
            return f"{column}_{suffix}"
    return column


def _to_millimetres(df, unit_map, scale):
    """Convert pixel columns to millimeters using each occurrence's px/mm scale.

    Lengths divide by the scale once and areas twice. Converted columns gain an `_mm` or
    `_mm2` suffix, and an occurrence with no calibration gets NaN.
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
        converted[column] = (
            f"{column}_{suffix}",
            pd.to_numeric(df[column], errors="coerce") / (scale**power),
        )

    if not converted:
        logger.warning(
            "units='mm' but no column is in px or px2 -- nothing to convert (units seen: %s)",
            sorted({unit_map.get(c) for c in df.columns if c in unit_map}),
        )
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
        logger.warning(
            "%d of %d occurrence(s) have no scale, so their %d converted column(s) are NaN",
            missing,
            len(out),
            len(converted),
        )
    if untouched:
        logger.info(
            "left %d non-length column(s) as they were: %s",
            len(untouched),
            ", ".join(sorted(untouched)[:5]) + ("..." if len(untouched) > 5 else ""),
        )

    return out


def _exported_name(column, unit, units):
    """Return the name a column of `unit` is exported under, suffixed where `units` converts it."""
    conversion = CONVERTIBLE_UNITS.get(unit) if units == "mm" else None
    return f"{column}_{conversion[0]}" if conversion else column


def _filter_only_rows(project_path, filters, present, units, current_only):
    """Return the stored values behind filter columns the export's selection left out.

    Args:
        project_path: Project to read from.
        filters: The filter columns wanted.
        present: Columns the export already has, under their exported names.
        units: As `export_metrics` takes it.
        current_only: As `export_metrics` takes it.

    Returns:
        `(long rows, their column provenance)`, or `(None, {})` where nothing more is needed
        or no run holds it.
    """
    wanted = {column for column in filters if column not in present}
    if not wanted:
        return None, {}

    # A column can't be split back into run, part and metric (a metric name may
    # hold "__" itself), so the run is found by prefix and the rest by building
    # that run's columns.
    stored = run_records.load_runs(project_path, kind="metric")
    names = (
        []
        if stored.empty
        else sorted(
            {
                name
                for name in stored["name"].dropna()
                if any(column.startswith(f"{name}__") for column in wanted)
            }
        )
    )
    if not names:
        return None, {}

    long_df = _current_long(project_path, run_names=names, current_only=current_only)
    behind = {
        (info["run_name"], info["part"], info["metric_name"])
        for column, info in _column_provenance(long_df).items()
        if _exported_name(column, info["unit"], units) in wanted
    }
    if not behind:
        return None, {}

    keys = zip(long_df["run_name"], long_df["part"], long_df["metric_name"])
    rows = long_df[[key in behind for key in keys]]
    return rows, _column_provenance(rows)


class FilteredWide(NamedTuple):
    """The wide frame after unit conversion and row filters, with what an export records about it.

    Attributes:
        df: One row per occurrence that passed, filter-only and transform-info columns still in it.
        long_df: The long rows behind the selected columns.
        provenance: `_column_provenance` of the selected columns, before unit conversion.
        filter_rows: Long rows read only for a filter, or None.
        hidden: Columns in `df` that were read for a filter or hold transform info, to drop before writing.
        metric_columns: The selected metric columns of `df`, under their converted names.
    """

    df: pd.DataFrame
    long_df: pd.DataFrame
    provenance: dict
    filter_rows: object
    hidden: set
    metric_columns: list


def filtered_wide(
    project_path,
    run_names=None,
    parts=None,
    metric_names=None,
    filters=None,
    filter_columns=(),
    occurrence_columns=None,
    subset=None,
    drop_empty=True,
    current_only=True,
    units=None,
    transform_info=False,
):
    """Build the wide frame an export starts from: occurrences joined to values, converted, then filtered.

    Units are converted before filters run, so a threshold in millimeters filters millimeters.

    Args:
        project_path: Project to read from.
        run_names: Run names to include; all if None.
        parts: Parts to include; all if None.
        metric_names: Metric names to include; all if None.
        filters: `{column: (op, value)}` or `{column: predicate}` (see `passing`). A column
            the selection left out is still read for the filter.
        filter_columns: Further columns to read where the selection left them out, for a
            filter the caller applies afterwards.
        occurrence_columns: Occurrence columns to join; all if None.
        subset: Named subset to restrict to.
        drop_empty: Drop occurrences with no metric value.
        current_only: Only values measured from the project's current masks.
        units: `"mm"` converts pixel columns using each occurrence's scale; None leaves pixels.
        transform_info: Keep the columns of what each metric run's transforms recorded.

    Returns:
        A `FilteredWide`.
    """
    long_df = _current_long(
        project_path, run_names=run_names, parts=parts, metric_names=metric_names, current_only=current_only
    )
    provenance = _column_provenance(long_df)

    occurrences = subset_selection.select_occurrences(
        project_path,
        subset=subset,
        columns=list(occurrence_columns) if occurrence_columns else None,
    )

    # A filter may name a column the selection left out: its values are loaded
    # for the filter and never exported. Nothing to look for when nothing was
    # narrowed, since every column is already here.
    filter_rows, filter_provenance = None, {}
    filter_columns = set(filters or {}) | set(filter_columns)
    if filter_columns and not (run_names is None and parts is None and metric_names is None):
        present = set(occurrences.columns) | {
            _exported_name(column, info["unit"], units) for column, info in provenance.items()
        }
        if units == "mm":
            present.add(scale_calibration.SCALE_COL)
        filter_rows, filter_provenance = _filter_only_rows(
            project_path, filter_columns, present, units, current_only
        )

    df = _wide_from_long(long_df if filter_rows is None else pd.concat([long_df, filter_rows]))

    # Transform-info columns stay in the frame until filters have run, so a
    # filter can target one, and are dropped after unless asked for.
    hidden = set()
    if not transform_info and metric_names is None:
        hidden = {column for column, info in provenance.items() if info["unit"] == TRANSFORM_INFO_UNIT}
        long_df = long_df[long_df["unit"] != TRANSFORM_INFO_UNIT]
        provenance = {column: info for column, info in provenance.items() if column not in hidden}
    hidden |= set(filter_provenance)
    metric_columns = [column for column in df.columns if column != ID_COL and column not in hidden]

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
        scale = scale_calibration.scale_for_occurrences(project_path, occurrence_ids=df[ID_COL])
        if scale.empty or not scale.notna().any():
            raise ValueError(
                "units='mm' but this project has no scale covering these "
                "occurrences, so every converted column would be empty. "
                "Record one with declare_scale() for a rig whose px/mm you "
                "know, or measure_scales() for images with a target in frame."
            )
        df = _to_millimetres(
            df,
            {column: info["unit"] for column, info in {**provenance, **filter_provenance}.items()},
            scale,
        )
        metric_columns = [converted_name(column, df.columns) for column in metric_columns]
        # a filter-only column was converted too, and is dropped under that name
        hidden = {converted_name(column, df.columns) for column in hidden}

    if filters:
        df = apply_filters(df, filters)

    return FilteredWide(df, long_df, provenance, filter_rows, hidden, metric_columns)
