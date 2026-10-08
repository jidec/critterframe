"""
Derived metrics: a value computed from one occurrence-part's own metric values, e.g. a ratio.

A derived metric sees only its occurrence's values; a value fit across a population is a group metric
(`metrics.outliers`).
"""

import inspect

from ..recipes import canonical_json
from .stored import StoredValueMetric


class DerivedMetric(StoredValueMetric):
    """
    Operation: `fn` applied to one occurrence-part's values, stored by `from_run` or measured earlier in this run.

    - `fn` -- a named module-level function `fn(values, **parameters) -> value`,
      `values` being `{metric_name: value}` for this occurrence alone. Its
      import path and `version` identify it in the recipe hash, so a lambda or
      a nested function is refused: neither has a stable name.
    - `features` -- Metric operations whose values `fn` receives. Each must be
      an operation of `from_run`'s current recipe, or with `from_run=None` a
      metric listed before this one in the same run, which may then be given
      by its metric name instead: `run_metrics` looks it up.
    - `from_run` -- the metric run holding those values, for the same part;
      None reads the metrics before this one in its own run.
    - `name` -- metric name; defaults to `fn`'s name.
    - `unit` -- recorded unit.
    - `version` -- bump by hand when `fn` changes what it returns.
    - `parameters` -- JSON-serializable keyword arguments for `fn`, in the recipe hash.
    """

    same_run = True

    def __init__(self, fn, features, from_run=None, name=None, unit=None, version="1",
                 parameters=None):
        _require_named_function(fn)
        super().__init__("derived", self._score, features, from_run,
                         version=version, unit=unit, metric_name=name or fn.__name__)
        # Features given by name, which bind() replaces with this run's own operations.
        self.unbound = [feature for feature in self.features if isinstance(feature, str)]
        if self.unbound and from_run is not None:
            raise ValueError(
                f"{self.metric_name}: feature(s) {self.unbound} are named, which only "
                f"reads this run's own metrics -- to read run {from_run!r}, pass the "
                "operations it measured with")
        self.fn = fn
        self.arguments = dict(parameters or {})
        try:
            canonical_json(self.arguments)
        except TypeError as exc:
            raise TypeError(f"derived() parameters must be JSON-serializable: {exc}") from None
        self._values = {}

    def bind(self, earlier):
        """
        This metric with each named feature replaced by the earlier metric of that name.

        - `earlier` -- the Metric operations listed before this one in the run.

        Returns itself when no feature is named, a new DerivedMetric otherwise.
        """
        if not self.unbound:
            return self
        by_name = {operation.metric_name: operation for operation in earlier}
        missing = [name for name in self.unbound if name not in by_name]
        if missing:
            raise ValueError(
                f"{self.metric_name} reads {missing} from this run, but no metric of "
                "that name is listed before it in metrics=")
        features = [by_name[feature] if isinstance(feature, str) else feature
                    for feature in self.features]
        return DerivedMetric(self.fn, features, name=self.metric_name, unit=self.unit,
                             version=self.version, parameters=self.arguments)

    def _require_bound(self):
        if self.unbound:
            raise ValueError(
                f"{self.metric_name} names its feature(s) {self.unbound} rather than "
                "holding them -- run it through run_metrics, which looks them up")

    def spec(self):
        self._require_bound()
        spec = super().spec()
        spec["parameters"] = {
            "function": f"{self.fn.__module__}.{self.fn.__qualname__}",
            "features": [feature.spec() for feature in self.features],
            "from_run": self.from_run,
        }
        # Only when given, so no hash recorded before parameters existed moves.
        if self.arguments:
            spec["parameters"]["arguments"] = self.arguments
        return spec

    def prepare(self, context):
        """
        Load the stored values of every occurrence this run covers; nothing with `from_run=None`.

        A lookup, not a fit: each value is computed from its own occurrence's
        row alone, so the record names the upstream recipe and no population.

        Returns `{"from_run", "from_recipe_hash", "features"}`.
        """
        self._require_bound()
        if self.from_run is None:
            return {"from_run": None, "from_recipe_hash": None,
                    "features": [feature.metric_name for feature in self.features]}

        from_recipe_hash = self.check_from_run(context)
        columns = self.feature_values(context)

        self._values = {}
        for name, series in columns.items():
            for occurrence_id, value in series.items():
                self._values.setdefault(occurrence_id, {})[name] = value

        return {
            "from_run": self.from_run,
            "from_recipe_hash": from_recipe_hash,
            "features": list(columns),
        }

    def _score(self, target):
        """`fn` of this occurrence's values; no input yet where any feature has no current value."""
        self._require_bound()
        occurrence_id = self.require_stored(target)
        if self.from_run is None:
            values = {feature.metric_name: target.values[feature.metric_name]
                      for feature in self.features if feature.metric_name in target.values}
        else:
            values = self._values.get(occurrence_id, {})
        if len(values) < len(self.features):
            raise self.no_value()
        return self.fn(dict(values), **self.arguments)


def _require_named_function(fn):
    if not inspect.isfunction(fn):
        raise TypeError(f"derived() needs a function, got {type(fn).__name__}")
    if fn.__name__ == "<lambda>" or "<locals>" in fn.__qualname__:
        raise ValueError(
            f"derived() needs a module-level function, not {fn.__qualname__!r} -- "
            "its import path is what identifies it in the recipe hash")


def derived(fn, features, from_run=None, **kwargs):
    """Operation: DerivedMetric, in the lowercase factory style of every other metric."""
    return DerivedMetric(fn, features, from_run, **kwargs)
