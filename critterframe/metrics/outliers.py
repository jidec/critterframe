"""Group metrics, fitted per group in prepare(): outlier(), cluster(), and their base classes."""

import logging
from functools import partial

import numpy as np
import pandas as pd

from ..recipes import Metric
from ..selectionhelpers import sample_occurrences
from .pixels import masked_pixels
from .stored import StoredValueMetric
from ..records import occurrences as occurrence_records
from ..records.occurrences import ID_COL, ids_record, load_occurrences
from ..visualization import figures, grids

logger = logging.getLogger(__name__)

# Key for the population-wide fallback model. Every occurrence whose group is
# unknown, or whose group is too small to fit its own model, is scored against
# this one rather than going unscored -- a missing species label should cost you
# precision, not the measurement.
POPULATION = None

# Groups with fewer reference occurrences than this don't get their own model.
# Fitting an outlier detector on three points produces confident nonsense.
MIN_GROUP_SIZE = 1

# Groups drawn by name in the reference figure; the rest are pooled as "(other)".
FIGURE_GROUPS = 10

# Members sampled per cluster in a cluster gallery.
GALLERY_MEMBERS = 8


def group_lookup(project_path, group_col, occurrence_ids=None):
    """Return `{occurrence_id: group}` read from the occurrence table.

    Args:
        project_path: Project to read from.
        group_col: Occurrence column naming the group. None returns an empty mapping, which
            puts every occurrence on the population-wide fit.
        occurrence_ids: Occurrences to read; all if None.
    """
    if not group_col:
        return {}

    occurrence_records.require_columns(project_path, group_col, "nothing to group by")

    groups = load_occurrences(project_path, columns=[group_col])
    if occurrence_ids is not None:
        groups = groups[groups[ID_COL].isin({str(i) for i in occurrence_ids})]
    # A missing value is left OUT rather than becoming a group of its own:
    # not knowing which group an occurrence belongs to is what the
    # population-wide fallback is for, and GroupMetric's own groupby already
    # drops it.
    groups = groups[groups[group_col].notna()]
    return groups.set_index(ID_COL)[group_col].to_dict()


def fitted_for(fits, group_by_id, occurrence_id):
    """Return `(fit, group)` for one occurrence: its group's fit, or the population-wide one.

    Args:
        fits: `{group: fit}`, including a `POPULATION` entry.
        group_by_id: `{occurrence_id: group}`, from `group_lookup`.
        occurrence_id: The occurrence being scored.
    """
    group = group_by_id.get(occurrence_id, POPULATION)
    fit = fits.get(group)
    if fit is None:
        group, fit = POPULATION, fits[POPULATION]
    return fit, group


class PooledPixelGroupMetric(Metric):
    """Base for a metric fitted per group on pixels pooled from the reference images.

    A subclass supplies `_fit(pixels)`, `_score(segment)` and its own `spec()`.

    Args:
        name: Operation name.
        score_fn: The function scoring one segment.
        group_col: Occurrence column to group by. None fits one population.
        transforms: Operations applied when gathering the reference pixels; pass the ones
            the run uses.
        min_group_size: Groups with fewer reference occurrences than this share the
            population-wide fit.
        sample_pixels: Pixels sampled per reference occurrence when pooling.
        reference: Fit against reference masks.
        metric_name: Name the value is stored under.
        unit: Recorded unit.
        version: Method version.
    """

    def __init__(
        self,
        name,
        score_fn,
        group_col=None,
        transforms=(),
        min_group_size=MIN_GROUP_SIZE,
        sample_pixels=2000,
        reference=False,
        metric_name=None,
        unit="category",
        version="1",
    ):
        super().__init__(name, score_fn, version=version, unit=unit, metric_name=metric_name or name)
        self.group_col = group_col
        self.transforms = list(transforms)
        self.min_group_size = min_group_size
        self.sample_pixels = sample_pixels
        self.reference = reference

        self.fits = {}
        self.group_by_id = {}

    def _fit(self, pixels):
        """Fit one group's model on its pooled BGR pixels. Implemented by the subclass."""
        raise NotImplementedError

    def _pixels(self, segment):
        """Return one segment's masked pixels for pooling, or None where there is no mask."""
        return masked_pixels(segment, cap=self.sample_pixels)

    def _describe_fit(self, group):
        """Return extra keys about one group's fit for the run record."""
        return {}

    def prepare(self, context):
        """Pool every reference occurrence's masked pixels per group, and fit each group.

        Returns:
            The fit record: each fit's contributing occurrences as a count and digest, with
            groups too small for their own fit marked `fitted=False`.
        """
        from ..segments import iterate_segments

        self.group_by_id = group_lookup(context.project_path, self.group_col, context.occurrence_ids)

        pooled = {}
        contributors = {}
        for occurrence_id, segment in iterate_segments(
            context.project_path,
            part=context.part,
            transforms=self.transforms,
            reference=self.reference,
            occurrence_ids=context.occurrence_ids,
        ):
            pixels = self._pixels(segment)
            if pixels is None:
                continue
            group = self.group_by_id.get(occurrence_id, POPULATION)
            for key in {group, POPULATION}:
                pooled.setdefault(key, []).append(pixels)
                contributors.setdefault(key, []).append(occurrence_id)

        if POPULATION not in pooled:
            raise ValueError(
                f"no reference occurrences with usable pixels for "
                f"{self.metric_name} -- segment this part first"
            )

        groups = {}
        for group, chunks in pooled.items():
            fitted = group is POPULATION or len(chunks) >= self.min_group_size
            if fitted:
                self.fits[group] = self._fit(np.concatenate(chunks))
            else:
                logger.warning(
                    "group %r has only %d reference occurrence(s) (< "
                    "min_group_size=%d) -- using the population-wide fit",
                    group,
                    len(chunks),
                    self.min_group_size,
                )
            if group is not POPULATION:
                record = dict(ids_record(contributors[group]), fitted=fitted)
                if fitted:
                    record.update(self._describe_fit(group))
                groups[str(group)] = record

        return {
            "group_col": self.group_col,
            "min_group_size": self.min_group_size,
            "reference": self.reference,
            "population": dict(ids_record(contributors[POPULATION]), **self._describe_fit(POPULATION)),
            "groups": groups,
        }

    def _fit_for(self, occurrence_id):
        """Return this occurrence's group's fit, or the population-wide one."""
        if not self.fits:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you; calling "
                "the operation directly skips it"
            )
        return fitted_for(self.fits, self.group_by_id, occurrence_id)


def _estimator_spec(model):
    """Return an sklearn model's configuration as JSON: class and scalar parameters, per pipeline step."""
    steps = getattr(model, "steps", None)
    if steps is not None:
        return {"pipeline": [_estimator_spec(step) for _name, step in steps]}
    params = {}
    if hasattr(model, "get_params"):
        params = {
            key: value
            for key, value in sorted(model.get_params(deep=False).items())
            if value is None or isinstance(value, (str, int, float, bool))
        }
    return {"class": type(model).__name__, "params": params}


class GroupMetric(StoredValueMetric):
    """Base for a metric fitted once per group over a population of stored values.

    The population is seen only by the fit in `prepare()`; each occurrence is then scored
    on its own stored row. A vector feature contributes one column per element.

    Args:
        features: Ordered metric operations making up the feature vector, e.g.
            `[body_length(), max_width()]`. Each must be an operation of `from_run`'s
            current recipe.
        from_run: The metric run holding those stored values.
        group_col: Occurrence column to group by, e.g. `"taxon"`. None fits one model.
        min_group_size: Groups smaller than this use the population-wide model.
        model_factory: Zero-argument callable returning an unfitted model with `.fit(X)`.
        score_fn: Callable `(model, features) -> dict`, `features` being a 1xN array.
        name: Operation name.
        metric_name: Name the value is stored under.
        unit: Recorded unit.
        version: Method version.
        default_model_factory: The subclass's default model. A configured model that
            differs from it reaches the hash.
    """

    def __init__(
        self,
        features,
        from_run,
        group_col=None,
        min_group_size=MIN_GROUP_SIZE,
        model_factory=None,
        score_fn=None,
        name=None,
        metric_name=None,
        unit="category",
        version="1",
        default_model_factory=None,
    ):
        if model_factory is None or score_fn is None:
            raise ValueError(
                f"{type(self).__name__} needs both a model_factory (zero-arg, "
                "returns an unfit model with .fit(X)) and a "
                "score_fn(model, features) -> dict"
            )
        if not features:
            raise ValueError("a group metric needs at least one feature")

        super().__init__(
            name or type(self).__name__.lower(),
            self._score,
            features,
            from_run,
            version=version,
            unit=unit,
            metric_name=metric_name or name or type(self).__name__.lower(),
        )

        self.group_col = group_col
        self.min_group_size = min_group_size
        self.model_factory = model_factory
        self.score_fn = score_fn
        self.default_model_factory = default_model_factory

        self.models = {}
        self.group_by_id = {}
        self.columns = []
        self._stored = {}
        self._fit_labels = {}

    def spec(self):
        """Return the operation's spec: features, reference run, grouping and model configuration."""
        spec = super().spec()
        parameters = {
            "features": [feature.spec() for feature in self.features],
            "from_run": self.from_run,
            "group_col": self.group_col,
            "min_group_size": self.min_group_size,
            "model": type(self.model_factory()).__name__,
        }
        # Only when it differs from the default, so every hash recorded before
        # model settings reached it holds.
        if self.default_model_factory is not None:
            configured = _estimator_spec(self.model_factory())
            if configured != _estimator_spec(self.default_model_factory()):
                parameters["model_config"] = configured
        spec["parameters"] = parameters
        return spec

    def prepare(self, context):
        """Fit the reference models over every occurrence the run covers.

        Also writes a figure of the reference population through `context.report`.

        Returns:
            The fit record: the reference run and its recipe, the features, the grouping, and
            each group's reference occurrences as a count and digest, with groups too small for
            their own model marked `fitted=False`.
        """
        from_recipe_hash = self.check_from_run(context)

        reference, columns = self.feature_table(context)
        self.columns = columns
        if reference.empty or not columns:
            raise ValueError(
                f"no reference values for {self.metric_name}: run "
                f"'{self.from_run}' has no stored "
                f"{[feature.metric_name for feature in self.features]} values "
                f"for part '{context.part}'. Run those metrics first -- a group "
                "metric scores against a population that has already been "
                "measured."
            )
        if self.group_col:
            groups = load_occurrences(context.project_path, columns=[self.group_col])
            reference = reference.merge(groups, on=ID_COL, how="left")

        self.models, self._fit_labels = {}, {}
        groups = {}
        if self.group_col:
            self.group_by_id = reference.set_index(ID_COL)[self.group_col].to_dict()
            for group, rows in reference.groupby(self.group_col):
                # Dropped by subset rather than by projection so the ids stay
                # beside the numbers: the same rows fit the model and name it.
                clean = rows.dropna(subset=columns)
                fitted = len(clean) >= self.min_group_size
                if fitted:
                    self._fit_group(group, clean)
                else:
                    logger.warning(
                        "group %r has only %d reference occurrences (< "
                        "min_group_size=%d) -- scoring it against the "
                        "population-wide model instead of its own",
                        group,
                        len(clean),
                        self.min_group_size,
                    )
                groups[str(group)] = dict(ids_record(clean[ID_COL]), fitted=fitted)

        population = reference.dropna(subset=columns)
        if population.empty:
            raise ValueError("no reference occurrences have every feature value populated")
        self._fit_group(POPULATION, population)
        self._stored = dict(zip(population[ID_COL], population[columns].to_numpy(dtype=float)))

        logger.info(
            "%s fit: %d group model(s) + 1 population-wide fallback, %d reference occurrences",
            self.metric_name,
            len(self.models) - 1,
            len(population),
        )

        if context.report:
            context.report.figure(
                f"{self.metric_name}__reference", self._reference_figure(population, columns)
            )

        return {
            "from_run": self.from_run,
            "from_recipe_hash": from_recipe_hash,
            "group_col": self.group_col,
            "features": [feature.metric_name for feature in self.features],
            "model": type(self.model_factory()).__name__,
            "min_group_size": self.min_group_size,
            "population": ids_record(population[ID_COL]),
            "groups": groups,
        }

    def _fit_group(self, key, rows):
        """Fit one group's model on its rows, keeping its fit labels where it has no `predict()`."""
        model = self._fit(rows[self.columns])
        self.models[key] = model
        if not hasattr(model, "predict") and hasattr(model, "labels_"):
            self._fit_labels[key] = dict(zip(rows[ID_COL], model.labels_))

    def _figure_labels(self, population):
        """Return a label per reference occurrence to color the reference figure by, or None."""
        if not self.group_col:
            return None
        labels = (
            population[self.group_col]
            .astype(object)
            .where(population[self.group_col].notna(), "(none)")
            .astype(str)
        )
        named = set(labels.value_counts().head(FIGURE_GROUPS).index)
        return labels.where(labels.isin(named), "(other)").tolist()

    def _reference_figure(self, population, columns):
        """Return a figure of the fitted population: a scatter, a histogram, or a 2-D PCA projection."""
        title = f"{self.metric_name}: reference from '{self.from_run}' (n={len(population)})"
        groups = self._figure_labels(population)

        if len(columns) > len(self.features) or len(columns) > 2:
            projected = _project_2d(population[columns].to_numpy(dtype=float))
            return figures.scatter(
                projected[:, 0], projected[:, 1], groups=groups, xlabel="PC1", ylabel="PC2", title=title
            )
        if len(columns) == 2:
            return figures.scatter(
                population[columns[0]],
                population[columns[1]],
                groups=groups,
                xlabel=columns[0],
                ylabel=columns[1],
                title=title,
            )
        values = population[columns[0]]
        if groups is not None:
            labels = pd.Series(groups, index=population.index)
            values = {group: rows.tolist() for group, rows in values.groupby(labels)}
        else:
            values = values.tolist()
        return figures.histogram(values, xlabel=columns[0], title=title)

    def _fit(self, frame):
        model = self.model_factory()
        model.fit(frame.to_numpy())
        return model

    def _model_for(self, occurrence_id):
        """Return this occurrence's group's model, or the population-wide one."""
        return fitted_for(self.models, self.group_by_id, occurrence_id)

    def _score_features(self, model, group, occurrence_id, features):
        """Return `score_fn`'s dict for one occurrence's feature row."""
        return self.score_fn(model, features)

    def _score(self, target):
        """Score one occurrence's stored row against its group's model.

        Returns:
            `score_fn`'s dict plus `group`, the group that scored it, None for the
            population-wide model.
        """
        occurrence_id = self.require_stored(target)
        if not self.models:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you; calling "
                "the operation directly skips it"
            )

        row = self._stored.get(occurrence_id)
        if row is None:
            raise self.no_value()
        model, group = self._model_for(occurrence_id)

        result = dict(self._score_features(model, group, occurrence_id, row[None, :]))
        result["group"] = group
        return result


def _project_2d(matrix):
    """Return rows projected onto their first two principal components, zero-padded when fewer exist."""
    centered = matrix - matrix.mean(axis=0)
    if len(centered) < 2:
        return np.zeros((len(centered), 2))
    _u, _s, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((len(projected), 1))])
    return projected


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
        from ..segments import iterate_segments

        assignments = self._assignments()
        largest = sorted(
            assignments, key=lambda group: -sum(len(ids) for ids in assignments[group].values())
        )[:FIGURE_GROUPS]
        shown = {
            group: {
                cluster_id: sample_occurrences(ids, GALLERY_MEMBERS)
                for cluster_id, ids in assignments[group].items()
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


def outlier(features, from_run, **kwargs):
    """Operation: an `OutlierMetric`.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        **kwargs: The other `OutlierMetric` arguments.
    """
    return OutlierMetric(features, from_run, **kwargs)


def cluster(features, from_run, **kwargs):
    """Operation: a `ClusterMetric`.

    Args:
        features: As in `GroupMetric`.
        from_run: As in `GroupMetric`.
        **kwargs: The other `ClusterMetric` arguments.
    """
    return ClusterMetric(features, from_run, **kwargs)
