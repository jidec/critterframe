"""Project-reading selections: which occurrences match stored values, have a result, or typify a group."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from ..project import paths
from ..core.recipes import DEFAULT_PART
from ..records import masks as mask_records
from ..records import metrics as metric_records
from ..records import runs as run_records
from ..records.occurrences import ID_COL, load_occurrences
from ..storage.imagestore import ImageStore
from ..wide import column_name, filtered_wide, metrics_wide
from . import subsets as subset_selection
from .algorithms import group_medoids, rows_matching

logger = logging.getLogger(__name__)


def ids_with_image(project_path):
    """Return the occurrences with an image in the store, as a sorted list of ids.

    Args:
        project_path: Project to read from.
    """
    if not paths.images_path(project_path).exists():
        return []
    with ImageStore(project_path, readonly=True) as store:
        return sorted(store.keys())


def ids_with_mask(project_path, part=DEFAULT_PART, reference=False):
    """Return the occurrences with a mask for a part, as a sorted list of ids.

    Args:
        project_path: Project to read from.
        part: The part.
        reference: Read the reference masks.
    """
    masks = mask_records.load_masks(
        project_path, parts=[part], reference=reference, columns=["occurrence_id", "part"]
    )
    return sorted(set(masks[ID_COL].astype(str))) if not masks.empty else []


def ids_matching(project_path, run_name, rules, part=DEFAULT_PART, current_only=False):
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


def ids_passing(
    project_path, filters, run_names=None, parts=None, subset=None, units=None, current_only=True
):
    """Return the occurrences an export with these filters would keep, as a sorted list of ids.

    An occurrence with no value in the selected columns never passes.

    Args:
        project_path: Project to read from.
        filters: `{column: (op, value)}` or `{column: predicate}`, by full column name, as
            `export_metrics` takes them.
        run_names: Run names whose columns count as measured; all if None.
        parts: Parts whose columns count as measured; all if None.
        subset: Named subset to restrict to.
        units: `"mm"` to write thresholds against converted columns; None for pixels.
        current_only: Only values measured from the project's current masks.
    """
    paths.require_project(project_path)
    built = filtered_wide(
        project_path,
        run_names=run_names,
        parts=parts,
        filters=filters,
        occurrence_columns=[ID_COL],
        subset=subset,
        current_only=current_only,
        units=units,
    )
    return sorted(built.df[ID_COL].astype(str))


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


def ids_completed(project_path, run_name, part=None, kind=None, reference=False):
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
