"""Clusters: the cluster an occurrence falls in within its own group, from stored values."""

from functools import partial

from ..records.occurrences import ID_COL
from ..selection.algorithms import sample_ids
from ..visualization import grids
from .base.group import FIGURE_GROUPS, MIN_GROUP_SIZE, POPULATION, GroupMetric

# Members sampled per cluster in a cluster gallery.
GALLERY_MEMBERS = 8


def _cluster_score(model, features, probability_threshold=None):
    """Return the cluster a feature row falls in, and how firmly where the model can say.

    Args:
        model: The fitted clusterer.
        features: The 1xN feature row.
        probability_threshold: Probability below which the cluster is reported as -1.
    """
    cluster_id = int(model.predict(features)[0])
    result = {"cluster_id": cluster_id}
    if hasattr(model, "transform"):
        result["centroid_distance"] = float(model.transform(features)[0][cluster_id])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][cluster_id])
        result["probability"] = probability
        if probability_threshold is not None and probability < probability_threshold:
            result["cluster_id"] = -1
    return result


def _default_kmeans(n_clusters=3):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0)


def _reduced(model, n_components):
    """Return `model` behind standardization and a PCA down to `n_components`."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0), model)


class ClusterMetric(GroupMetric):
    """Group metric: the cluster an occurrence falls in within its own group, and how firmly.

    Cluster ids are comparable only within one group. A clusterer without `predict()`
    (DBSCAN) reports its own fit labels, with -1 as noise.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        group_col: As in `GroupMetric`.
        n_clusters: Clusters per group for the default KMeans. Ignored with `model_factory`.
        min_group_size: As in `GroupMetric`.
        model_factory: As in `GroupMetric`.
        name: Name the value is stored under. The operation stays `"cluster"`.
        unit: Recorded unit.
        n_components: Standardize the features and reduce them by PCA to this many
            dimensions before clustering.
        probability_threshold: With a probabilistic model, a cluster probability below
            this is reported as cluster -1.
    """

    def __init__(
        self,
        features,
        from_run,
        group_col=None,
        n_clusters=3,
        min_group_size=MIN_GROUP_SIZE,
        model_factory=None,
        name=None,
        unit="category",
        n_components=None,
        probability_threshold=None,
    ):
        base_factory = model_factory or (lambda: _default_kmeans(n_clusters))
        factory = base_factory if n_components is None else lambda: _reduced(base_factory(), n_components)
        if probability_threshold is not None and not hasattr(factory(), "predict_proba"):
            raise ValueError(
                "probability_threshold needs a model with predict_proba(), "
                "e.g. model_factory=lambda: GaussianMixture(3)"
            )

        super().__init__(
            features,
            from_run,
            group_col=group_col,
            min_group_size=min_group_size,
            model_factory=factory,
            score_fn=partial(_cluster_score, probability_threshold=probability_threshold),
            name="cluster",
            metric_name=name,
            unit=unit,
            default_model_factory=_default_kmeans,
        )
        self.probability_threshold = probability_threshold

    def spec(self):
        """Return the operation's spec, with the PCA step and probability threshold where they are set."""
        spec = super().spec()
        if self.probability_threshold is not None:
            spec["parameters"]["probability_threshold"] = self.probability_threshold
        return spec

    def _score_features(self, model, group, occurrence_id, features):
        labels = self._fit_labels.get(group)
        if labels is not None:
            return {"cluster_id": int(labels[occurrence_id])}
        return super()._score_features(model, group, occurrence_id, features)

    def _assignments(self):
        """Return `{group: {cluster_id: [occurrence ids]}}` over the reference population."""
        assignments = {}
        for occurrence_id, row in self._stored.items():
            model, group = self._model_for(occurrence_id)
            cluster_id = self._score_features(model, group, occurrence_id, row[None, :])["cluster_id"]
            assignments.setdefault(group, {}).setdefault(cluster_id, []).append(occurrence_id)
        return assignments

    def _figure_labels(self, population):
        if self.group_col:
            return super()._figure_labels(population)
        labels = []
        for occurrence_id, row in zip(population[ID_COL], population[self.columns].to_numpy(dtype=float)):
            model, group = self._model_for(occurrence_id)
            labels.append(
                f"cluster {self._score_features(model, group, occurrence_id, row[None, :])['cluster_id']}"
            )
        return labels

    def prepare(self, context):
        """Fit the reference models, then draw a gallery of sampled members per cluster."""
        record = super().prepare(context)
        if context.report:
            self._draw_galleries(context)
        return record

    def _draw_galleries(self, context):
        """Draw one grid per group, for the largest groups: a row of sampled members per cluster."""
        from ..core.segments import iterate_segments

        assignments = self._assignments()
        largest = sorted(
            assignments, key=lambda group: -sum(len(ids) for ids in assignments[group].values())
        )[:FIGURE_GROUPS]
        shown = {
            group: {
                cluster_id: sample_ids(ids, GALLERY_MEMBERS) for cluster_id, ids in assignments[group].items()
            }
            for group in largest
        }
        wanted = sorted(
            {
                occurrence_id
                for clusters in shown.values()
                for ids in clusters.values()
                for occurrence_id in ids
            }
        )

        cells = {}
        for occurrence_id, segment in iterate_segments(
            context.project_path,
            part=context.part,
            transforms=context.transforms,
            reference=context.reference,
            occurrence_ids=wanted,
            progress=f"{self.metric_name} gallery",
        ):
            cells[occurrence_id] = grids.mask_cutout(segment)
        if not cells:
            return

        for group in largest:
            clusters = sorted(shown[group])
            rows = [
                [cells[occurrence_id] for occurrence_id in shown[group][cluster_id] if occurrence_id in cells]
                for cluster_id in clusters
            ]
            if not any(rows):
                continue
            labels = [f"{cluster_id} n={len(assignments[group][cluster_id])}" for cluster_id in clusters]
            title = f"{self.metric_name}: clusters of '{self.from_run}'"
            suffix = ""
            if group is not POPULATION:
                title += f", group {group}"
                suffix = f"__{group}"
            grid = grids.comparison_grid(rows, row_labels=labels, title=title)
            context.report.figure(f"{self.metric_name}__clusters{suffix}", grid)


def cluster(features, from_run, **kwargs):
    """Operation: a `ClusterMetric`.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        **kwargs: The other `ClusterMetric` arguments.
    """
    return ClusterMetric(features, from_run, **kwargs)
