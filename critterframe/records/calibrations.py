"""Calibration records: the scope and provenance every calibration type shares, with an opaque payload."""

import logging
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..core.recipes import canonical_json, load_json
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, load_occurrences
from ..storage.tables import load_table, upsert_table

logger = logging.getLogger(__name__)

TYPE_COL = "calibration_type"
KEY_COLS = [TYPE_COL, "scope", "scope_value"]

COLUMNS = [
    TYPE_COL,
    "scope",
    "scope_value",
    "parameters_json",
    "source",
    "score",
    "measured_from",
    "created_at",
]


def make_calibration_row(
    calibration_type, scope, scope_value, parameters, source, score=None, measured_from=None
):
    """Build one calibration record.

    Args:
        calibration_type: The kind, e.g. `"scale"`. Part of the key.
        scope: Occurrence column identifying what the record covers, e.g. `"event_id"`, or
            `ID_COL` for one occurrence.
        scope_value: The value in that column.
        parameters: JSON-serializable dict of whatever the type needs; never interpreted here.
        source: How it was obtained: `"target"`, `"declared"`, or an extension's own name.
        score: Quality of the measurement, e.g. a template match's correlation peak.
        measured_from: What it was measured on: an image key, a file name, a note.
    """
    if not isinstance(parameters, dict):
        raise TypeError(
            f"calibration parameters must be a dict, got "
            f"{type(parameters).__name__} -- even a single-number calibration "
            "is stored as one, so that adding a second number later doesn't "
            "change the shape of the table"
        )

    return {
        TYPE_COL: str(calibration_type),
        "scope": str(scope),
        "scope_value": str(scope_value),
        "parameters_json": canonical_json(parameters),
        "source": source,
        "score": None if score is None else float(score),
        "measured_from": measured_from,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_calibrations(project_path, rows):
    """Write calibration records, replacing any with the same `(type, scope, scope_value)`.

    Args:
        project_path: Project to write to.
        rows: Records from `make_calibration_row`.
    """
    if not rows:
        return 0

    upsert_table(
        pd.DataFrame(rows, columns=COLUMNS), paths.calibrations_path(project_path), key_cols=KEY_COLS
    )
    return len(rows)


def load_calibrations(project_path, calibration_type=None, scope=None):
    """Read the calibration table, with `parameters` parsed into dicts.

    Args:
        project_path: Project to read from.
        calibration_type: Kind to narrow to.
        scope: Scope column to narrow to.

    Returns:
        A DataFrame; empty if nothing has been calibrated.
    """
    df = load_table(paths.calibrations_path(project_path), missing_ok=True)
    if df.empty:
        return pd.DataFrame(columns=[c for c in COLUMNS if c != "parameters_json"] + ["parameters"])

    if calibration_type is not None:
        df = df[df[TYPE_COL] == calibration_type]
    if scope is not None:
        df = df[df["scope"] == scope]

    df = df.reset_index(drop=True)
    df["parameters"] = [load_json(value) for value in df.pop("parameters_json")]
    return df


def require_scope_column(project_path, scope):
    """Raise unless the occurrence table has the scope column."""
    occurrence_records.require_columns(project_path, scope, "nothing to key a calibration on")
    return scope


def pending_scope_values(project_path, calibration_type, scope, max_new=None):
    """Return the values of an occurrence column with no calibration of this type yet.

    Args:
        project_path: Project to read.
        calibration_type: Kind of calibration, e.g. `"scale"`.
        scope: Occurrence column the calibration is keyed on.
        max_new: Cap on the values returned, applied after calibrated ones are excluded.
    """
    require_scope_column(project_path, scope)
    occurrences = load_occurrences(project_path, columns=[scope])

    values = occurrences[scope].dropna().astype(str).unique()
    measured = set(
        load_calibrations(project_path, calibration_type=calibration_type, scope=scope)["scope_value"].astype(
            str
        )
    )
    pending = [value for value in values if value not in measured]

    return pending[:max_new] if max_new is not None else pending


def _occurrences_per_value(occurrences, scope):
    """Return how many occurrences one value of a scope covers on average: how broad the scope is."""
    distinct = occurrences[scope].astype(str).nunique()
    return len(occurrences) / distinct if distinct else float("inf")


def resolve_for_occurrences(project_path, calibration_type, occurrence_ids=None):
    """Return the calibration parameters that apply to each occurrence.

    Where several records could apply, one scoped to `ID_COL` wins; otherwise the scope
    covering the fewest occurrences wins, with a warning.

    Args:
        project_path: Project to read from.
        calibration_type: Kind of calibration.
        occurrence_ids: Occurrences to resolve; all if None.

    Returns:
        A Series of parameter dicts indexed by occurrence id, None where nothing applies.
    """
    calibrations = load_calibrations(project_path, calibration_type=calibration_type)
    if calibrations.empty:
        return pd.Series(dtype="object", name=calibration_type)

    scope_columns = list(dict.fromkeys(calibrations["scope"]))
    occurrences = load_occurrences(project_path)

    missing = [s for s in scope_columns if s not in occurrences.columns]
    if missing:
        logger.warning(
            "%s calibration rows are scoped on column(s) the "
            "occurrence table doesn't have, so they apply to "
            "nothing: %s",
            calibration_type,
            ", ".join(sorted(missing)),
        )
        scope_columns = [s for s in scope_columns if s not in missing]

    if occurrence_ids is not None:
        wanted = {str(occurrence_id) for occurrence_id in occurrence_ids}
        occurrences = occurrences[occurrences[ID_COL].isin(wanted)]

    # Broadest scope first, most specific last, so each pass overwrites the one
    # before it and the narrowest statement is what survives.
    ordered = sorted(
        scope_columns, key=lambda scope: (scope == ID_COL, -_occurrences_per_value(occurrences, scope))
    )

    index = occurrences[ID_COL].astype(str)
    resolved = pd.Series([None] * len(index), index=index, dtype="object", name=calibration_type)

    for scope in ordered:
        rows = calibrations[calibrations["scope"] == scope]
        lookup = dict(zip(rows["scope_value"].astype(str), rows["parameters"]))

        values = occurrences[scope].astype(str).map(lookup)
        values.index = index

        overridden = int((values.notna() & resolved.notna()).sum())
        if overridden:
            logger.warning(
                "%d occurrence(s) already had a %s calibration from "
                "a broader scope; '%s' is more specific and overrides "
                "it",
                overridden,
                calibration_type,
                scope,
            )

        resolved = values.combine_first(resolved)

    logger.info(
        "resolved a %s calibration for %d of %d occurrence(s) from scope(s): %s",
        calibration_type,
        int(resolved.notna().sum()),
        len(resolved),
        ", ".join(ordered) or "none",
    )
    return resolved
