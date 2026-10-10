"""Export: the wide one-row-per-occurrence trait table written as a file, and its manifest."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .calibrations import scale as scale_calibration
from .project import paths
from .core.recipes import hash_spec, recorded_callable
from .records import calibrations as calibration_records
from .records import runs as run_records
from .records.occurrences import ID_COL, ids_record
from .storage.jsonfiles import append_jsonl, read_jsonl, write_json
from .wide import converted_name, filtered_wide, metric_units, passing

logger = logging.getLogger(__name__)


def _blank_failing_parts(df, part_filters, part_columns):
    """Empty each part's columns on the rows failing that part's filters.

    Every part is judged before any is emptied, so a filter reading another part's column
    sees the value as measured.

    Args:
        df: The frame.
        part_filters: `{part: filters}`, each as `passing` takes them.
        part_columns: `{part: its columns in df}`.

    Returns:
        `(df, counts, blanked)`: `counts` is `{part: {kept, filtered_out, not_measured}}`
        and `blanked` is `{part: occurrence ids emptied}`.
    """
    failing = {}
    for part, filters in part_filters.items():
        measured = df[part_columns[part]].notna().any(axis=1)
        failing[part] = (measured & ~passing(df, filters), measured)

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
        provenance: `wide._column_provenance` of that frame, before unit conversion.
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
        final = converted_name(column, default_columns)
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
        filters: `{column: (op, value)}` or `{column: predicate}` (see `passing`). A
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

    part_filters = dict(part_filters or {})
    built = filtered_wide(
        project_path,
        run_names=run_names,
        parts=parts,
        metric_names=metric_names,
        filters=filters,
        filter_columns={column for conditions in part_filters.values() for column in conditions},
        occurrence_columns=occurrence_columns,
        subset=subset,
        drop_empty=drop_empty,
        current_only=current_only,
        units=units,
        transform_info=transform_info,
    )
    df, long_df, provenance = built.df, built.long_df, built.provenance
    filter_rows, hidden, metric_columns = built.filter_rows, built.hidden, built.metric_columns

    part_counts = {}
    if part_filters:
        part_columns = {
            part: [
                exported
                for column, info in provenance.items()
                if info["part"] == part and (exported := converted_name(column, df.columns)) in df.columns
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
