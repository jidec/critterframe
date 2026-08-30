"""
The scope/provenance machinery every calibration type shares.

A calibration is knowledge about the imaging system -- how big a pixel is, how
a camera's colours relate to true ones. It describes equipment and conditions,
never an organism, so it can't live on the occurrence table.

    calibration_type scope         scope_value    parameters
    scale            event_id      12835          {"px_per_mm": 11.83}

The scope and provenance are generic; the payload is not. `parameters` is an
opaque JSON dict this layer never interprets, because a scale is one number
while a colour correction is a method, a matrix, an offset and an illuminant.
Each calibration type owns a module under calibration/ that supplies the
meaning.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from ..project import paths
from ..recipes import canonical_json, load_json
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


def make_calibration_row(calibration_type, scope, scope_value, parameters,
                         source, score=None, measured_from=None):
    """
    Build one calibration record.

    The three key fields are coerced to str here, because storage compares keys
    by value without coercing (see CLAUDE.md).

    calibration_type -- what kind this is: "scale", "color". Part of the key, so
                        two kinds can describe one session without colliding.
    scope            -- occurrence column identifying what this covers, e.g.
                        "event_id", "device", or ID_COL for one occurrence.
    scope_value      -- the value in that column this applies to.
    parameters       -- dict of whatever the type needs, stored as JSON and never
                        interpreted here, so a type can grow a field without a
                        schema change. Must be JSON-serializable.
    source           -- how it was obtained: "target" (measured against a
                        reference of known size), "declared" (stated by someone
                        who knows the rig), or an extension's own name. A measured
                        calibration and an asserted one deserve different trust.
    score            -- quality of the measurement where one exists, e.g. a
                        template match's correlation peak.
    measured_from    -- what it was measured on: an image key, a filename, a note.
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
    """
    Write calibration rows, replacing any existing row for the same
    (type, scope, scope_value).

    Upsert rather than append: like a mask, the current calibration is the one
    that counts, and a re-measurement supersedes rather than accumulates. Unlike
    a metric there's no result to keep the history of -- what a corrected
    calibration invalidates is nothing, because nothing was ever stored in
    converted units.
    """
    if not rows:
        return 0

    upsert_table(pd.DataFrame(rows, columns=COLUMNS),
                 paths.calibrations_path(project_path), key_cols=KEY_COLS)
    return len(rows)


def load_calibrations(project_path, calibration_type=None, scope=None):
    """
    Read the calibration table, with `parameters` parsed back into dicts.

    Missing is fine and means nothing has been calibrated yet -- an empty frame
    rather than a raise, since a project measuring only shape traits or only
    relative colour never needs a calibration at all.
    """
    df = load_table(paths.calibrations_path(project_path), missing_ok=True)
    if df.empty:
        return pd.DataFrame(columns=[c for c in COLUMNS
                                     if c != "parameters_json"] + ["parameters"])

    if calibration_type is not None:
        df = df[df[TYPE_COL] == calibration_type]
    if scope is not None:
        df = df[df["scope"] == scope]

    df = df.reset_index(drop=True)
    df["parameters"] = [load_json(value) for value in df.pop("parameters_json")]
    return df


def require_scope_column(project_path, scope):
    """
    Raise unless the occurrence table has this column, naming the ones it does.

    A scope IS an occurrence column (see the module docstring), so this is
    records.occurrences.require_columns asking on a calibration's behalf --
    named here because "is this a usable scope" is the question a caller of this
    module is actually asking.
    """
    occurrence_records.require_columns(
        project_path, scope, "nothing to key a calibration on")
    return scope


def pending_scope_values(project_path, calibration_type, scope, limit=None):
    """
    Values of one occurrence column with no calibration of this type yet -- the
    repeat-aware check a measurement pass makes before doing any work, so
    measuring is resumable and re-running it is a no-op.
    """
    require_scope_column(project_path, scope)
    occurrences = load_occurrences(project_path, columns=[scope])

    values = occurrences[scope].dropna().astype(str).unique()
    measured = set(load_calibrations(project_path,
                                     calibration_type=calibration_type,
                                     scope=scope)["scope_value"].astype(str))
    pending = [value for value in values if value not in measured]

    return pending[:limit] if limit is not None else pending


def _occurrences_per_value(occurrences, scope):
    """
    How many occurrences one value of this scope covers, on average -- the
    measure of how BROAD a scope is, used to order specificity.

    A device column with two values across 5,000 occurrences scores 2,500; a
    session column with 60 values scores 83; occurrence_id scores 1. Ordering by
    it needs no hardcoded hierarchy of column names, so a project inventing its
    own scope gets sensible precedence without telling anyone about it.
    """
    distinct = occurrences[scope].astype(str).nunique()
    return len(occurrences) / distinct if distinct else float("inf")


def resolve_for_occurrences(project_path, calibration_type, occurrence_ids=None):
    """
    The calibration parameters that apply to each occurrence, as a Series of
    dicts indexed by occurrence_id.

    An occurrence with no applicable row gets None -- never a project-wide
    average, never the nearest session's value. A missing calibration has to
    stay missing, because a trait converted with a guessed one is
    indistinguishable in a CSV from a trait converted with a measured one.

    Precedence, where more than one row could apply:

      1. A row scoped to ID_COL wins. It describes that occurrence and nothing
         else, so it is the most specific statement available -- which is what
         makes "a target in this particular frame" override "the session this
         frame belongs to".
      2. Otherwise the scope covering the FEWEST occurrences wins, on the same
         reasoning: a calibration measured per deployment says more about one
         night than one measured per device says about a season. A warning fires
         when this happens, because two overlapping calibrations usually means
         one was meant to replace the other rather than join it.
    """
    calibrations = load_calibrations(project_path,
                                     calibration_type=calibration_type)
    if calibrations.empty:
        return pd.Series(dtype="object", name=calibration_type)

    scope_columns = list(dict.fromkeys(calibrations["scope"]))
    occurrences = load_occurrences(project_path)

    missing = [s for s in scope_columns if s not in occurrences.columns]
    if missing:
        logger.warning("%s calibration rows are scoped on column(s) the "
                       "occurrence table doesn't have, so they apply to "
                       "nothing: %s", calibration_type, ", ".join(sorted(missing)))
        scope_columns = [s for s in scope_columns if s not in missing]

    if occurrence_ids is not None:
        wanted = {str(occurrence_id) for occurrence_id in occurrence_ids}
        occurrences = occurrences[occurrences[ID_COL].isin(wanted)]

    # Broadest scope first, most specific last, so each pass overwrites the one
    # before it and the narrowest statement is what survives.
    ordered = sorted(scope_columns,
                     key=lambda scope: (scope == ID_COL,
                                        -_occurrences_per_value(occurrences, scope)))

    index = occurrences[ID_COL].astype(str)
    resolved = pd.Series([None] * len(index), index=index, dtype="object",
                         name=calibration_type)

    for scope in ordered:
        rows = calibrations[calibrations["scope"] == scope]
        lookup = dict(zip(rows["scope_value"].astype(str), rows["parameters"]))

        values = occurrences[scope].astype(str).map(lookup)
        values.index = index

        overridden = int((values.notna() & resolved.notna()).sum())
        if overridden:
            logger.warning("%d occurrence(s) already had a %s calibration from "
                           "a broader scope; '%s' is more specific and overrides "
                           "it", overridden, calibration_type, scope)

        resolved = values.combine_first(resolved)

    logger.info("resolved a %s calibration for %d of %d occurrence(s) from "
                "scope(s): %s", calibration_type, int(resolved.notna().sum()),
                len(resolved), ", ".join(ordered) or "none")
    return resolved
