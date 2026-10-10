"""Derived metrics: a value computed from one occurrence-part's own metric values."""

import inspect

from ..core.recipes import canonical_json
from .base.stored import StoredValueMetric


class DerivedMetric(StoredValueMetric):
    """Operation: a function applied to one occurrence-part's metric values.

    Args:
        fn: A module-level function `fn(values, **parameters) -> value`, where `values` is
            `{metric_name: value}` for one occurrence. Lambdas and nested functions are
            refused: its import path identifies it in the recipe hash.
        features: The metric operations whose values `fn` receives. With `from_run=None`
            they may be given as metric names, looked up among the metrics listed before
            this one.
        from_run: The metric run holding those values; None reads the metrics before this
            one in its own run.
        name: Metric name; `fn`'s name if None.
        unit: Recorded unit.
        version: Bumped by hand when `fn` changes what it returns.
        parameters: JSON-serializable keyword arguments for `fn`; hashed.
    """

    same_run = True

    def __init__(self, fn, features, from_run=None, name=None, unit=None, version="1", parameters=None):
        _require_named_function(fn)
        super().__init__(
            "derived",
            self._score,
            features,
            from_run,
            version=version,
            unit=unit,
            metric_name=name or fn.__name__,
        )
        # Features given by name, which bind() replaces with this run's own operations.
        self.unbound = [feature for feature in self.features if isinstance(feature, str)]
        if self.unbound and from_run is not None:
            raise ValueError(
                f"{self.metric_name}: feature(s) {self.unbound} are named, which only "
                f"reads this run's own metrics -- to read run {from_run!r}, pass the "
                "operations it measured with"
            )
        self.fn = fn
        self.arguments = dict(parameters or {})
        try:
            canonical_json(self.arguments)
        except TypeError as exc:
            raise TypeError(f"derived() parameters must be JSON-serializable: {exc}") from None
        self._values = {}

    def bind(self, earlier):
        """Return this metric with each named feature replaced by the earlier metric of that name.

        Args:
            earlier: The metric operations listed before this one in the run.
        """
        if not self.unbound:
            return self
        by_name = {operation.metric_name: operation for operation in earlier}
        missing = [name for name in self.unbound if name not in by_name]
        if missing:
            raise ValueError(
                f"{self.metric_name} reads {missing} from this run, but no metric of "
                "that name is listed before it in metrics="
            )
        features = [by_name[feature] if isinstance(feature, str) else feature for feature in self.features]
        return DerivedMetric(
            self.fn,
            features,
            name=self.metric_name,
            unit=self.unit,
            version=self.version,
            parameters=self.arguments,
        )

    def _require_bound(self):
        if self.unbound:
            raise ValueError(
                f"{self.metric_name} names its feature(s) {self.unbound} rather than "
                "holding them -- run it through run_metrics, which looks them up"
            )

    def spec(self):
        """Return the operation's spec, with the function's import path, its features and `from_run`."""
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
        """Load the stored values this run's occurrences need; nothing with `from_run=None`.

        Returns:
            `{"from_run", "from_recipe_hash", "features"}`.
        """
        self._require_bound()
        if self.from_run is None:
            return {
                "from_run": None,
                "from_recipe_hash": None,
                "features": [feature.metric_name for feature in self.features],
            }

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
        """Return `fn` of this occurrence's values, raising `NoInput` where a feature has none."""
        self._require_bound()
        occurrence_id = self.require_stored(target)
        if self.from_run is None:
            values = {
                feature.metric_name: target.values[feature.metric_name]
                for feature in self.features
                if feature.metric_name in target.values
            }
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
            "its import path is what identifies it in the recipe hash"
        )


def derived(fn, features, from_run=None, **kwargs):
    """Operation: a function applied to one occurrence-part's metric values.

    Args:
        fn: As in `DerivedMetric`.
        features: As in `DerivedMetric`.
        from_run: As in `DerivedMetric`.
        **kwargs: The other `DerivedMetric` arguments.
    """
    return DerivedMetric(fn, features, from_run, **kwargs)
