"""
Derived metrics: a value computed from one occurrence-part's own stored metric values, e.g. a ratio.

A derived metric sees only its occurrence's values; a value fit across a population is a group metric
(`metrics.outliers`).
"""

import inspect

from .stored import StoredValueMetric


class DerivedMetric(StoredValueMetric):
    """
    Operation: `fn` applied to one occurrence-part's stored values from `from_run`.

    - `fn` -- a named module-level function `fn(values) -> value`, `values`
      being `{metric_name: stored value}` for this occurrence alone. Its import
      path and `version` identify it in the recipe hash, so a lambda or a
      nested function is refused: neither has a stable name.
    - `features` -- Metric operations whose stored values `fn` receives. Each
      must be an operation of `from_run`'s current recipe.
    - `from_run` -- the metric run holding those values, for the same part.
    - `name` -- metric name; defaults to `fn`'s name.
    - `unit` -- recorded unit.
    - `version` -- bump by hand when `fn` changes what it returns.
    """

    def __init__(self, fn, features, from_run, name=None, unit=None, version="1"):
        _require_named_function(fn)
        super().__init__("derived", self._score, features, from_run,
                         version=version, unit=unit, metric_name=name or fn.__name__)
        self.fn = fn
        self._values = {}

    def spec(self):
        spec = super().spec()
        spec["parameters"] = {
            "function": f"{self.fn.__module__}.{self.fn.__qualname__}",
            "features": [feature.spec() for feature in self.features],
            "from_run": self.from_run,
        }
        return spec

    def prepare(self, context):
        """
        Load the stored values of every occurrence this run covers.

        A lookup, not a fit: each value is computed from its own occurrence's
        row alone, so the record names the upstream recipe and no population.

        Returns `{"from_run", "from_recipe_hash", "features"}`.
        """
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
        occurrence_id = self.require_stored(target)
        values = self._values.get(occurrence_id, {})
        if len(values) < len(self.features):
            raise self.no_value()
        return self.fn(dict(values))


def _require_named_function(fn):
    if not inspect.isfunction(fn):
        raise TypeError(f"derived() needs a function, got {type(fn).__name__}")
    if fn.__name__ == "<lambda>" or "<locals>" in fn.__qualname__:
        raise ValueError(
            f"derived() needs a module-level function, not {fn.__qualname__!r} -- "
            "its import path is what identifies it in the recipe hash")


def derived(fn, features, from_run, **kwargs):
    """Operation: DerivedMetric, in the lowercase factory style of every other metric."""
    return DerivedMetric(fn, features, from_run, **kwargs)
