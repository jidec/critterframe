"""Stored-value metrics: what a derived or group metric reads instead of a segment."""

import logging

import numpy as np
import pandas as pd

from ..drivers import NoInput
from ..recipes import Metric, hash_spec
from ..records import runs as run_records
from ..records.metrics import latest_values
from ..records.occurrences import ID_COL

logger = logging.getLogger(__name__)


class StoredValues:
    """What a stored-input metric is scored on: an occurrence-part, and its run's values so far.

    Args:
        occurrence_id: Occurrence being scored.
        part: Part being scored.
        values: `{metric_name: value}` of the metrics before this one in the same run.
    """

    def __init__(self, occurrence_id, part, values=None):
        self.occurrence_id = str(occurrence_id)
        self.part = part
        self.values = dict(values or {})

    def __repr__(self):
        return f"StoredValues({self.occurrence_id!r}, {self.part!r})"


def is_vector(value):
    """Return whether a stored value is a list, tuple or array."""
    return isinstance(value, (list, tuple, np.ndarray))


def feature_columns(metric_name, values):
    """Return one stored feature as numeric columns.

    A scalar feature is one column; a vector is `<name>_<i>` per element. A vector whose
    length differs from the most common one becomes NaN, with a warning.

    Args:
        metric_name: The feature's name, the column prefix.
        values: Series of stored values indexed by occurrence id.
    """
    if values.empty:
        return pd.DataFrame(index=values.index)
    if not any(is_vector(value) for value in values):
        return pd.DataFrame({metric_name: pd.to_numeric(values, errors="coerce")})

    lengths = values.map(lambda value: len(value) if is_vector(value) else -1)
    width = int(lengths[lengths >= 0].mode().iloc[0])
    wrong = lengths != width
    if wrong.any():
        logger.warning(
            "%d stored '%s' value(s) are not %d-long vectors -- left out",
            int(wrong.sum()),
            metric_name,
            width,
        )
    rows = [
        np.full(width, np.nan) if bad else np.asarray(value, dtype=float) for value, bad in zip(values, wrong)
    ]
    return pd.DataFrame(
        np.vstack(rows), index=values.index, columns=[f"{metric_name}_{index}" for index in range(width)]
    )


class StoredValueMetric(Metric):
    """Base for a metric computed from stored values instead of a segment.

    A subclass supplies `prepare()` and the function scoring one `StoredValues`.

    Args:
        name: As in `Metric`.
        function: As in `Metric`.
        features: The metric operations whose stored values are read. Each must be an
            operation of `from_run`'s current recipe.
        from_run: The metric run holding those values. None, where a subclass sets
            `same_run`, reads the metrics before this one in its own run.
        **kwargs: The other `Metric` arguments.
    """

    input = "stored"
    # Whether from_run=None (read this run's earlier metrics) is meaningful. A
    # population fit happens in prepare(), before this run has measured anything.
    same_run = False

    def __init__(self, name, function, features, from_run, **kwargs):
        if not features:
            raise ValueError(f"{name} needs at least one feature")
        if from_run is None and not self.same_run:
            raise ValueError(
                f"{name} needs from_run: it fits on another run's stored "
                "values before this run measures anything"
            )
        super().__init__(name, function, **kwargs)
        self.features = list(features)
        self.from_run = from_run

    def bind(self, earlier):
        """Return this metric with its features resolved against the metrics before it in its run.

        Args:
            earlier: The metric operations listed before this one, already bound.
        """
        return self

    def check_from_run(self, context):
        """Check every feature is an operation of `from_run`'s current recipe.

        Returns:
            That recipe's hash, or None where `from_run` has never run.
        """
        recipe_hash, recipe_spec = run_records.current_recipe(
            context.project_path, self.from_run, context.part
        )
        if recipe_spec is None:
            return recipe_hash

        stored = {hash_spec(operation) for operation in recipe_spec.get("operations", [])}
        unmatched = [
            feature.metric_name for feature in self.features if hash_spec(feature.spec()) not in stored
        ]
        if unmatched:
            raise ValueError(
                f"{self.metric_name}: feature(s) {unmatched} aren't configured "
                f"as in run {self.from_run!r}'s current recipe ({recipe_hash}) "
                f"for part {context.part!r} -- pass the same operations that "
                "run measured with, so the values are the ones they describe"
            )
        return recipe_hash

    def feature_values(self, context, occurrence_ids=None):
        """Return `{metric_name: Series}` of current stored values.

        Args:
            context: The run's `RunContext`.
            occurrence_ids: Occurrences to read; the run's own if None.
        """
        wanted = set(context.occurrence_ids if occurrence_ids is None else occurrence_ids)
        values = {}
        for feature in self.features:
            series = latest_values(
                context.project_path,
                self.from_run,
                part=context.part,
                metric_name=feature.metric_name,
                occurrence_ids=wanted,
            )
            values[feature.metric_name] = series[series.index.isin(wanted)]
        return values

    def feature_table(self, context, occurrence_ids=None):
        """Return the features as one table, a row per occurrence.

        Args:
            context: The run's `RunContext`.
            occurrence_ids: Occurrences to read; the run's own if None.

        Returns:
            `(table, columns)`: an `occurrence_id` column plus one numeric column per feature
            or per element of a vector feature, and those columns' names.
        """
        frames = [
            feature_columns(name, series)
            for name, series in self.feature_values(context, occurrence_ids).items()
        ]
        table = pd.concat(frames, axis=1)
        columns = list(table.columns)
        table.index.name = ID_COL
        return table.reset_index(), columns

    def require_stored(self, target):
        """Return the occurrence id of `target`, refusing anything but a `StoredValues`."""
        if not isinstance(target, StoredValues):
            raise TypeError(
                f"{self.metric_name} reads stored values, so it is scored on "
                f"StoredValues, not a {type(target).__name__} -- run it through "
                "run_metrics"
            )
        return target.occurrence_id

    def no_value(self):
        """Return the `NoInput` to raise where a feature has no current value."""
        if self.from_run is None:
            return NoInput(f"no value earlier in this run for {self.metric_name}")
        return NoInput(f"no current '{self.from_run}' value")
