"""Export: the wide one-row-per-occurrence trait table, its manifest, and selections read from stored values."""

import logging
import operator
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrations import scale as scale_calibration
from .project import paths, subsets as subset_selection
from .recipes import DEFAULT_PART, hash_spec, recorded_callable
from .records import calibrations as calibration_records
from .records import masks as mask_records
from .records import metrics as metric_records
from .records import runs as run_records
from .records.metrics import TRANSFORM_INFO_UNIT, load_metrics
from .records.occurrences import ID_COL, ids_record, load_occurrences
from .selectionhelpers import group_medoids, rows_matching
from .storage.jsonfiles import append_jsonl, read_jsonl, write_json

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


# Pixel units and what they become once a px/mm scale is applied: the suffix a
# converted column takes, and the power of the scale it's divided by. A length
# divides once, an area twice. Anything not listed -- a fraction, a category, an
# embedding, a laplacian variance -- has no physical length in it to convert and
# is left exactly as it is.
CONVERTIBLE_UNITS = {"px": ("mm", 1), "px2": ("mm2", 2)}


def _converted_name(column, columns):
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


def _apply_filters(df, filters):
    """Narrow a frame to the rows passing every condition.

    Args:
        df: The frame.
        filters: As `_passing` takes them.
    """
    keep = _passing(df, filters)
    logger.info("filters kept %d of %d occurrences", int(keep.sum()), len(df))
    return df[keep]


def _passing(df, filters):
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


def _blank_failing_parts(df, part_filters, part_columns):
    """Empty each part's columns on the rows failing that part's filters.

    Every part is judged before any is emptied, so a filter reading another part's column
    sees the value as measured.

    Args:
        df: The frame.
        part_filters: `{part: filters}`, each as `_passing` takes them.
        part_columns: `{part: its columns in df}`.

    Returns:
        `(df, counts, blanked)`: `counts` is `{part: {kept, filtered_out, not_measured}}`
        and `blanked` is `{part: occurrence ids emptied}`.
    """
    failing = {}
    for part, filters in part_filters.items():
        measured = df[part_columns[part]].notna().any(axis=1)
        failing[part] = (measured & ~_passing(df, filters), measured)

    df = df.copy()
    counts, blanked = {}, {}
    for part, (fails, measured) in failing.items():
        for column in part_columns[part]:
            df[column] = df[column].where(~fails)
        blanked[part] = set(df.loc[fails, ID_COL].astype(str))
        counts[part] = {
            "kept": int((measured & ~fails).sum()),
            "filtered_out": int(fails.sum()),
            "not_measured": int((~measured).sum()),
        }
        logger.info(
            "part '%s': filters kept %d of %d measured occurrence(s)",
            part,
            counts[part]["kept"],
            int(measured.sum()),
        )
    return df, counts, blanked


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


def _apply_rename(df, rename):
    """Relabel exported columns, changing no value and no column order.

    Args:
        df: The export.
        rename: `{exported column: new name}`.

    Raises:
        KeyError: If a column to rename isn't in the export.
        ValueError: If a new name collides with another column, or `occurrence_id` is renamed.
    """
    if ID_COL in rename:
        raise ValueError(f"'{ID_COL}' can't be renamed -- it is what every export is keyed by")

    missing = sorted(column for column in rename if column not in df.columns)
    if missing:
        raise KeyError(f"rename column(s) {missing} not in the export (available: {sorted(df.columns)})")

    targets = [str(name) for name in rename.values()]
    repeated = sorted({name for name in targets if targets.count(name) > 1})
    if repeated:
        raise ValueError(f"rename gives more than one column the name(s) {repeated}")

    kept = set(df.columns) - set(rename)
    taken = sorted(kept & set(targets))
    if taken:
        raise ValueError(f"rename target(s) {taken} are already columns of the export")

    return df.rename(columns={column: str(name) for column, name in rename.items()})


def occurrences_matching(project_path, run_name, rules, part=DEFAULT_PART, current_only=False):
    """Return the occurrences whose stored metric values match a `{metric: values}` rule set.

    An occurrence matches when any rule does, and a missing value never matches.

    Args:
        project_path: Project to read from.
        run_name: Run whose values the rules are written against.
        rules: `{metric_name: value}` or `{metric_name: [values, ...]}`, by bare metric name.
        part: Part the values were recorded for.
        current_only: Only values measured from the current masks. False by default,
            since a label that describes the image stays true after a resegmentation.

    Returns:
        A sorted list of occurrence ids; empty, with a warning, if the run has no values.

    Raises:
        KeyError: If a rule names a metric the run has no column for.
    """
    rules = {column_name(run_name, part, metric_name): values for metric_name, values in rules.items()}

    # transform_info on: a rule may name one, e.g. {"orient__unreliable": [True]},
    # and only the columns the rules name are read.
    df = metrics_wide(
        project_path, run_names=[run_name], parts=[part], current_only=current_only, transform_info=True
    )
    if df.empty:
        logger.warning(
            "run '%s' has no stored values for part '%s' -- nothing to match %s against, selecting none",
            run_name,
            part,
            sorted(rules),
        )
        return []

    matched = df[rows_matching(df, rules)]
    logger.info("%d of %d occurrence(s) in run '%s' match %s", len(matched), len(df), run_name, rules)
    return sorted(matched[ID_COL].astype(str))


def _is_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and not pd.isna(value)
    )


def _feature_vector(value):
    """Return one stored metric value as `(shape, vector)`, or None where it holds no number.

    A list is itself, a number a 1-long vector, and a dict its numeric values in sorted key
    order. `shape` is what two values must share to be compared: the length, or for a dict
    the keys used.
    """
    if isinstance(value, dict):
        keys = tuple(sorted(key for key, entry in value.items() if _is_number(entry)))
        if not keys:
            return None
        return keys, np.asarray([value[key] for key in keys], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value), np.asarray(value, dtype=float)
    if _is_number(value):
        return 1, np.asarray([value], dtype=float)
    return None


def exemplars_per_group(
    project_path,
    run_name,
    group_col,
    part=DEFAULT_PART,
    metric_name=None,
    count=1,
    occurrence_ids=None,
    normalize=True,
):
    """Return, per group, the occurrences most typical of it: the group's medoid in a stored feature.

    The medoid is the member with the smallest summed distance to the rest of its group.

    Args:
        project_path: Project to read from.
        run_name: The metric run holding the feature, e.g. an embedding run.
        group_col: Occurrence column naming the group, e.g. `"species"`. An occurrence
            with no value for it is excluded.
        part: Part the feature was measured on.
        metric_name: The feature; `run_name` if None. A vector, a number, or a dict of
            numbers such as a `threshold_fractions` result.
        count: How many per group.
        occurrence_ids: Only these occurrences: who can be chosen and who each is measured
            against.
        normalize: Scale each vector to unit length first. A single number is never
            scaled; pass False for values already on one scale, e.g. fractions.

    Returns:
        A sorted list of occurrence ids.
    """
    occurrences = load_occurrences(project_path)
    if group_col not in occurrences.columns:
        raise KeyError(f"no '{group_col}' column to group by (columns: {sorted(occurrences.columns)})")

    metric_name = run_name if metric_name is None else metric_name
    values = metric_records.latest_values(
        project_path, run_name, part=part, metric_name=metric_name, occurrence_ids=occurrence_ids
    )
    if occurrence_ids is not None:
        values = values[values.index.isin({str(i) for i in occurrence_ids})]
    if values.empty:
        logger.warning(
            "run '%s' has no current '%s' values for part '%s' -- no exemplars", run_name, metric_name, part
        )
        return []

    features = {occurrence_id: _feature_vector(value) for occurrence_id, value in values.items()}
    features = {occurrence_id: feature for occurrence_id, feature in features.items() if feature is not None}
    shapes = Counter(shape for shape, _vector in features.values())
    common = shapes.most_common(1)[0][0] if shapes else None
    usable = {occurrence_id: vector for occurrence_id, (shape, vector) in features.items() if shape == common}
    if len(usable) < len(values):
        logger.warning(
            "%d stored '%s' value(s) aren't shaped like the rest -- left out",
            len(values) - len(usable),
            metric_name,
        )
    if not usable:
        return []

    # A single number has no direction: scaled to unit length, every member
    # of a group would be the same point.
    if normalize and len(next(iter(usable.values()))) > 1:
        lengths = {occurrence_id: float(np.linalg.norm(vector)) for occurrence_id, vector in usable.items()}
        usable = {
            occurrence_id: vector / lengths[occurrence_id] if lengths[occurrence_id] else vector
            for occurrence_id, vector in usable.items()
        }

    groups = occurrences.set_index(ID_COL)[group_col]
    exemplars = group_medoids(usable, groups, count=count)
    logger.info(
        "%d exemplar(s) of '%s' by '%s' among %d occurrence(s)",
        len(exemplars),
        group_col,
        metric_name,
        len(usable),
    )
    return exemplars


def completed_ids(project_path, run_name, part=None, kind=None, reference=False):
    """Return the occurrences a run has a current result for.

    For a segmentation run, those whose current mask for the part was made by one of the
    run's recipes; for a metric run, those with a current value. An imported mask is not
    attributed to its import run.

    Args:
        project_path: Project to read from.
        run_name: The run.
        part: The part; optional where the run covers exactly one.
        kind: `"segment"` or `"metric"`, where both kinds share the name.
        reference: For a segmentation run, read the reference masks.

    Returns:
        A sorted list of occurrence ids still in the occurrence table.
    """
    runs = run_records.load_runs(project_path, name=run_name)
    if runs.empty:
        known = sorted(set(run_records.load_runs(project_path)["name"]))
        raise KeyError(f"no run named {run_name!r} -- this project has {known}")

    kinds = sorted(set(runs["kind"]))
    if kind is None:
        if len(kinds) > 1:
            raise ValueError(
                f"{run_name!r} names both a segmentation and a metric run -- say "
                "which with kind='segment' or kind='metric'"
            )
        kind = kinds[0]
    elif kind not in kinds:
        raise KeyError(f"no {kind} run named {run_name!r} -- it is a {kinds[0]} run")
    runs = runs[runs["kind"] == kind]

    parts = sorted(set(runs["part"]))
    if part is None:
        if len(parts) > 1:
            raise ValueError(f"run {run_name!r} covers parts {parts} -- say which with part=")
        part = parts[0]
    elif part not in parts:
        raise KeyError(f"run {run_name!r} has no part {part!r} -- it covers {parts}")

    if kind == "segment":
        hashes = set(runs.loc[runs["part"] == part, "recipe_hash"])
        masks = mask_records.load_masks(
            project_path, parts=[part], reference=reference, columns=["occurrence_id", "part", "recipe_hash"]
        )
        done = set() if masks.empty else set(masks.loc[masks["recipe_hash"].isin(hashes), ID_COL].astype(str))
    else:
        done = set(metric_records.result_keys(project_path, run_name, part)[ID_COL].astype(str))

    present = set(subset_selection.select_ids(project_path))
    return sorted(done & present)


def _plain(value):
    """Return one pandas cell as a JSON-serializable value, NaN and NaT as None."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _recorded_filters(filters):
    """Return the filters an export applied in storable form, a predicate by its name."""
    recorded = {}
    for column, condition in (filters or {}).items():
        if callable(condition):
            recorded[column] = {"callable": recorded_callable(condition)}
            continue
        op, value = condition
        if isinstance(value, (set, frozenset)):
            value = sorted(value, key=str)
        elif isinstance(value, tuple):
            value = list(value)
        recorded[column] = [op, value]
    return recorded


def _calibration_record(project_path, resolved):
    """Return the calibration behind an mm export: how many rows it covered, by scope and source.

    Args:
        project_path: Project to read from.
        resolved: The px/mm applied to each exported row.
    """
    rows = calibration_records.load_calibrations(
        project_path, calibration_type=scale_calibration.CALIBRATION_TYPE
    )
    covered = int(resolved.notna().sum())

    def counts(column):
        if rows.empty or column not in rows.columns:
            return {}
        return {str(value): int(n) for value, n in rows[column].value_counts().items()}

    return {
        "type": scale_calibration.CALIBRATION_TYPE,
        "covered": covered,
        "uncovered": int(len(resolved) - covered),
        "scopes": counts("scope"),
        "sources": counts("source"),
    }


def _export_record(
    project_path, df, long_df, provenance, selection, path=None, rename=None, part_counts=None
):
    """Return the record of what an export is, with its `export_hash`.

    Args:
        project_path: Project exported from.
        df: The finished export.
        long_df: The long frame it was built from.
        provenance: `_column_provenance` of that frame, before unit conversion.
        selection: The arguments that chose its rows and columns.
        path: Where the export was written, if it was.
        rename: The `{default name: exported name}` applied to `df`.
        part_counts: What `part_filters` did to each part.
    """
    values = long_df[long_df[ID_COL].astype(str).isin(set(df[ID_COL].astype(str)))]

    rename = {column: str(name) for column, name in (rename or {}).items()}
    default_of = {name: column for column, name in rename.items()}
    default_columns = [default_of.get(column, column) for column in df.columns]
    scale_col = rename.get(scale_calibration.SCALE_COL, scale_calibration.SCALE_COL)

    columns = {}
    for column, info in provenance.items():
        final = _converted_name(column, default_columns)
        if final not in default_columns:
            continue
        exported = rename.get(final, final)
        columns[exported] = dict(info, source_unit=info["unit"])
        if final != column:
            columns[exported]["unit"] = final[len(column) + 1 :]
        # The default name, so a relabelled column still says which
        # run__part__metric column it is.
        if exported != final:
            columns[exported]["column"] = final

    stored = run_records.load_runs(project_path)
    by_id = {row["run_id"]: row for row in stored.to_dict("records")} if not stored.empty else {}
    per_run = values.groupby("run_id").size() if not values.empty else pd.Series(dtype="int64")
    runs = []
    for run_id in sorted(per_run.index):
        row = by_id.get(run_id, {})
        runs.append(
            {
                "run_id": int(run_id),
                "name": _plain(row.get("name")),
                "kind": _plain(row.get("kind")),
                "part": _plain(row.get("part")),
                "subset": _plain(row.get("subset")),
                "recipe_hash": _plain(row.get("recipe_hash")),
                "recipe": row.get("recipe"),
                "context": row.get("context"),
                "created_at": _plain(row.get("created_at")),
                "n_values": int(per_run[run_id]),
            }
        )

    source_hashes = values["source_mask_hash"] if not values.empty else pd.Series(dtype="object")
    record = {
        "occurrences": ids_record(df[ID_COL]),
        "runs": runs,
        "columns": columns,
        # The derivations themselves, not just how many: a value's
        # source_mask_hash IS the identity of the segmentation behind it, so
        # listing them is what makes a resegmentation visible in the manifest.
        # Bounded by the number of segmentation recipes, not by occurrences.
        "source_masks": {
            "n_values": int(len(values)),
            "derivations": sorted(source_hashes.dropna().unique().tolist()),
            "n_without_provenance": int(source_hashes.isna().sum()),
        },
        "selection": selection,
        "calibration": (
            _calibration_record(project_path, df[scale_col]) if scale_col in df.columns else None
        ),
    }
    # Only when part filters ran, so an export made without them keeps its hash.
    if part_counts:
        record["part_counts"] = part_counts

    record["export_hash"] = hash_spec(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Relative when written inside the project, so a copied project's log still
    # names its own files; a CSV handed out elsewhere stays absolute.
    record["path"] = None if path is None else paths.relative_to_project(project_path, path)
    record["project"] = str(paths.project_dir(project_path))
    return record


def _write_manifest(project_path, record, path=None):
    """Write the record beside the export and append it to the project's exports log.

    Args:
        project_path: Project exported from.
        record: The record from `_export_record`.
        path: The exported file, or None where none was written.
    """
    if path is not None:
        sidecar = write_json(paths.export_sidecar_path(path), record)
        logger.info("wrote export manifest -> %s", sidecar)

    append_jsonl(paths.exports_log_path(project_path), record)
    return record


def load_exports(project_path):
    """Return every export the project has produced, oldest first, as a DataFrame."""
    return read_jsonl(paths.exports_log_path(project_path), what="export")


def export_metrics(
    project_path,
    path=None,
    run_names=None,
    parts=None,
    metric_names=None,
    filters=None,
    occurrence_columns=None,
    subset=None,
    drop_empty=True,
    current_only=True,
    units=None,
    manifest=True,
    transform_info=False,
    rename=None,
    part_filters=None,
):
    """Build the wide, one-row-per-occurrence trait table, write it as CSV, and return it.

    Units are converted before filters run, so a threshold in millimeters filters
    millimeters. `filters` drops rows; `part_filters` then empties the parts that fail.

    Args:
        project_path: Project to export from.
        path: CSV to write. None writes a uniquely named file under the project's
            `exports/` folder; False writes no CSV.
        run_names: Run names to include; all if None.
        parts: Parts to include; all if None.
        metric_names: Metric names to include; all if None.
        filters: `{column: (op, value)}` or `{column: predicate}` (see `_passing`). A
            column the selection left out is still read for the filter, and not exported.
        occurrence_columns: Occurrence columns to join; all if None.
        subset: Named subset to restrict to.
        drop_empty: Drop occurrences with no metric value.
        current_only: Only values measured from the project's current masks.
        units: `"mm"` converts pixel columns using each occurrence's scale; None leaves pixels.
        manifest: Write the export's manifest beside the file and into the exports log.
        transform_info: Include the columns of what each metric run's transforms recorded.
        rename: `{exported column: new name}`, applied last, so filters name the default
            columns.
        part_filters: `{part: filters}`. An occurrence failing a part's filters keeps its
            row with that part's columns emptied; the manifest counts each part's kept,
            filtered-out and unmeasured occurrences.

    Returns:
        The exported DataFrame.
    """
    paths.require_project(project_path)

    if path is False:
        path = None
    elif path is None:
        path = paths.default_export_path(project_path)

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
    part_filters = dict(part_filters or {})
    filter_columns = set(filters or {}) | {
        column for conditions in part_filters.values() for column in conditions
    }
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
        metric_columns = [_converted_name(column, df.columns) for column in metric_columns]
        # a filter-only column was converted too, and is dropped under that name
        hidden = {_converted_name(column, df.columns) for column in hidden}

    if filters:
        df = _apply_filters(df, filters)

    part_counts = {}
    if part_filters:
        part_columns = {
            part: [
                exported
                for column, info in provenance.items()
                if info["part"] == part and (exported := _converted_name(column, df.columns)) in df.columns
            ]
            for part in part_filters
        }
        nothing = sorted(part for part, columns in part_columns.items() if not columns)
        if nothing:
            raise ValueError(
                f"part_filters names part(s) {nothing} with no exported column to "
                f"empty -- this export holds {sorted({info['part'] for info in provenance.values()})}"
            )
        df, part_counts, blanked = _blank_failing_parts(df, part_filters, part_columns)
        if drop_empty and metric_columns:
            df = df[df[metric_columns].notna().any(axis=1)]
        # The manifest counts only the values that are in the file.
        emptied = [
            str(occurrence_id) in blanked.get(part, ())
            for occurrence_id, part in zip(long_df[ID_COL], long_df["part"])
        ]
        long_df = long_df[[not gone for gone in emptied]]

    df = df.drop(columns=[column for column in hidden if column in df.columns])
    if filter_rows is not None:
        # The manifest names the runs that decided the rows, as well as the
        # ones whose columns were exported.
        long_df = pd.concat([long_df, filter_rows])

    # occurrence_id first, then joined metadata, then the traits -- so the
    # identifying columns are on the left where anyone opening the CSV expects.
    ordered = (
        [ID_COL]
        + [c for c in df.columns if c not in metric_columns and c != ID_COL]
        + [c for c in metric_columns if c in df.columns]
    )
    df = df[ordered].reset_index(drop=True)

    if rename:
        df = _apply_rename(df, rename)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("exported %d occurrences x %d columns -> %s", len(df), len(df.columns), path)

    # After the CSV, so a manifest that cannot be written never costs the data.
    if manifest:
        _write_manifest(
            project_path,
            _export_record(
                project_path,
                df,
                long_df,
                provenance,
                selection={
                    # Key stays "runs": it is inside the export hash, so
                    # renaming it would make one table exported before and after
                    # this read as two different exports.
                    "runs": None if run_names is None else sorted(run_names),
                    "parts": None if parts is None else sorted(parts),
                    "metric_names": (None if metric_names is None else sorted(metric_names)),
                    "occurrence_columns": (
                        None if occurrence_columns is None else sorted(occurrence_columns)
                    ),
                    "subset": subset,
                    "filters": _recorded_filters(filters),
                    "drop_empty": bool(drop_empty),
                    "current_only": bool(current_only),
                    "units": units,
                    # Recorded only when on, so an export made before this option
                    # existed keeps the hash it was written under.
                    **({"transform_info": True} if transform_info else {}),
                    **({"rename": {column: str(name) for column, name in rename.items()}} if rename else {}),
                    **(
                        {
                            "part_filters": {
                                part: _recorded_filters(conditions)
                                for part, conditions in part_filters.items()
                            }
                        }
                        if part_filters
                        else {}
                    ),
                },
                path=path,
                rename=rename,
                part_counts=part_counts,
            ),
            path=path,
        )

    return df


def export_units(project_path, run_names=None):
    """Return the unit behind each exported column, as `{column: unit}`.

    Args:
        project_path: Project to read from.
        run_names: Run names to include; all if None.
    """
    return metric_units(project_path, run_names=run_names)
