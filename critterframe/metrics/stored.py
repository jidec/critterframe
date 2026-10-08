"""
Metrics over stored values: what a derived or group metric reads instead of a segment.

`run_metrics` hands a stored-input metric a `StoredValues`, never a Segment, so it cannot measure pixels.
"""

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
    """
    What a stored-input metric is scored on: which occurrence-part, and the values its own run measured so far.

    A metric reading another run looks its values up from what its `prepare()` loaded.

    - `occurrence_id` -- occurrence being scored.
    - `part` -- part being scored.
    - `values` -- `{metric_name: value}` of the metrics before this one in the
      same run, for this occurrence; what a derived metric with `from_run=None` reads.
    """

    def __init__(self, occurrence_id, part, values=None):
        self.occurrence_id = str(occurrence_id)
        self.part = part
        self.values = dict(values or {})

    def __repr__(self):
        return f"StoredValues({self.occurrence_id!r}, {self.part!r})"


def is_vector(value):
    return isinstance(value, (list, tuple, np.ndarray))


def feature_columns(metric_name, values):
    """
    One stored feature as numeric columns: itself when scalar, `<name>_<i>` per element when a vector.

    A vector whose length differs from the most common one is left out (all NaN) with a warning.

    - `metric_name` -- the feature's name, the column prefix.
    - `values` -- Series of stored values indexed by occurrence id.

    Returns a DataFrame indexed like `values`.
    """
    if values.empty:
        return pd.DataFrame(index=values.index)
    if not any(is_vector(value) for value in values):
        return pd.DataFrame({metric_name: pd.to_numeric(values, errors="coerce")})

    lengths = values.map(lambda value: len(value) if is_vector(value) else -1)
    width = int(lengths[lengths >= 0].mode().iloc[0])
    wrong = lengths != width
    if wrong.any():
        logger.warning("%d stored '%s' value(s) are not %d-long vectors -- left out",
                       int(wrong.sum()), metric_name, width)
    rows = [np.full(width, np.nan) if bad else np.asarray(value, dtype=float)
            for value, bad in zip(values, wrong)]
    return pd.DataFrame(np.vstack(rows), index=values.index,
                        columns=[f"{metric_name}_{index}" for index in range(width)])


class StoredValueMetric(Metric):
    """
    Base for a metric computed from another run's stored values rather than from a segment.

    Owns the reading: which run, which features, whether they are still what
    the features say they are. A subclass supplies `prepare()` and the
    function scoring one `StoredValues`.

    - `features` -- Metric operations whose stored values are read. Each must
      be an operation of `from_run`'s current recipe.
    - `from_run` -- the metric run holding those values; None, where a subclass
      sets `same_run`, for the metrics before this one in its own run.
    """

    input = "stored"
    # Whether from_run=None (read this run's earlier metrics) is meaningful. A
    # population fit happens in prepare(), before this run has measured anything.
    same_run = False

    def __init__(self, name, function, features, from_run, **kwargs):
        if not features:
            raise ValueError(f"{name} needs at least one feature")
        if from_run is None and not self.same_run:
            raise ValueError(f"{name} needs from_run: it fits on another run's stored "
                             "values before this run measures anything")
        super().__init__(name, function, **kwargs)
        self.features = list(features)
        self.from_run = from_run

    def bind(self, earlier):
        """
        This metric with its features resolved against the metrics before it in its run.

        - `earlier` -- the Metric operations listed before this one, already bound.

        Returns the metric to run: itself, unless a subclass names features to look up.
        """
        return self

    def check_from_run(self, context):
        """
        The recipe hash `from_run` currently designates for this part, after
        checking every feature is one of that recipe's operations.

        Returns the hash, or None where `from_run` has never run.
        """
        recipe_hash, recipe_spec = run_records.current_recipe(
            context.project_path, self.from_run, context.part)
        if recipe_spec is None:
            return recipe_hash

        stored = {hash_spec(operation) for operation in recipe_spec.get("operations", [])}
        unmatched = [feature.metric_name for feature in self.features
                     if hash_spec(feature.spec()) not in stored]
        if unmatched:
            raise ValueError(
                f"{self.metric_name}: feature(s) {unmatched} aren't configured "
                f"as in run {self.from_run!r}'s current recipe ({recipe_hash}) "
                f"for part {context.part!r} -- pass the same operations that "
                "run measured with, so the values are the ones they describe")
        return recipe_hash

    def feature_values(self, context, occurrence_ids=None):
        """
        `{metric_name: Series}` of current stored values.

        - `occurrence_ids` -- the occurrences to read; the run's own if None.
        """
        wanted = set(context.occurrence_ids if occurrence_ids is None else occurrence_ids)
        values = {}
        for feature in self.features:
            series = latest_values(context.project_path, self.from_run,
                                   part=context.part, metric_name=feature.metric_name,
                                   occurrence_ids=wanted)
            values[feature.metric_name] = series[series.index.isin(wanted)]
        return values

    def feature_table(self, context, occurrence_ids=None):
        """
        One row per occurrence with an `occurrence_id` column and one numeric
        column per feature, or per element of a vector feature.

        - `occurrence_ids` -- the occurrences to read; the run's own if None.

        Returns `(table, columns)`.
        """
        frames = [feature_columns(name, series)
                  for name, series in self.feature_values(context, occurrence_ids).items()]
        table = pd.concat(frames, axis=1)
        columns = list(table.columns)
        table.index.name = ID_COL
        return table.reset_index(), columns

    def require_stored(self, target):
        """The occurrence id of `target`, refusing anything but `StoredValues`."""
        if not isinstance(target, StoredValues):
            raise TypeError(
                f"{self.metric_name} reads stored values, so it is scored on "
                f"StoredValues, not a {type(target).__name__} -- run it through "
                "run_metrics")
        return target.occurrence_id

    def no_value(self):
        if self.from_run is None:
            return NoInput(f"no value earlier in this run for {self.metric_name}")
        return NoInput(f"no current '{self.from_run}' value")
