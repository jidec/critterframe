"""Outliers: whether an occurrence is unusual within its own group, from stored values."""

from .base.group import MIN_GROUP_SIZE, GroupMetric


def _isolation_forest_score(model, features):
    """Return `is_outlier` and `anomaly_score` for one feature row.

    `anomaly_score` is IsolationForest's decision function, where lower is more anomalous.
    """
    return {
        "is_outlier": bool(model.predict(features)[0] == -1),
        "anomaly_score": float(model.decision_function(features)[0]),
    }


def _default_isolation_forest(contamination="auto"):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(contamination=contamination, random_state=0)


class OutlierMetric(GroupMetric):
    """Group metric: whether an occurrence is unusual within its own group, and by how much.

    Uses IsolationForest unless `model_factory` is given.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        group_col: As in `GroupMetric`.
        min_group_size: As in `GroupMetric`.
        contamination: Expected fraction of outliers per group, passed to IsolationForest.
            Ignored with `model_factory`.
        model_factory: As in `GroupMetric`.
        name: Name the value is stored under. The operation stays `"outlier"`.
        unit: Recorded unit.
    """

    def __init__(
        self,
        features,
        from_run,
        group_col=None,
        min_group_size=MIN_GROUP_SIZE,
        contamination="auto",
        model_factory=None,
        name=None,
        unit="category",
    ):
        model_factory = model_factory or (lambda: _default_isolation_forest(contamination))
        super().__init__(
            features,
            from_run,
            group_col=group_col,
            min_group_size=min_group_size,
            model_factory=model_factory,
            score_fn=_isolation_forest_score,
            name="outlier",
            metric_name=name,
            unit=unit,
            default_model_factory=_default_isolation_forest,
        )


def outlier(features, from_run, **kwargs):
    """Operation: an `OutlierMetric`.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        **kwargs: The other `OutlierMetric` arguments.
    """
    return OutlierMetric(features, from_run, **kwargs)
