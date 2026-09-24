"""
Group metrics: outlier(), cluster().

"Is this specimen unusual?" isn't answerable from one segment -- it needs a
reference population, and usually the population of THIS species, since a body
length unremarkable for one species is impossible for another.

A group metric stores, exports, and composes like any other metric. The only
difference is that it fits once before the run's per-occurrence loop, via
prepare(), then scores each occurrence against its group's fitted model.
Reference values are read from an earlier metric run rather than recomputed.
"""

import logging
from functools import partial

import numpy as np
import pandas as pd

from ..maskops import mask_bounds
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
MIN_GROUP_SIZE = 5

# Groups drawn by name in the reference figure; the rest are pooled as "(other)".
FIGURE_GROUPS = 10

# Members sampled per cluster in a cluster gallery.
GALLERY_MEMBERS = 8


def group_lookup(project_path, group_col, occurrence_ids=None):
    """
    {occurrence_id: group} read off the occurrence table.

    Shared rather than reimplemented per group metric: "which group does this
    occurrence belong to, with a population-wide fallback when the answer is
    missing" is the same question for every metric fit per group, whether it
    fits on stored trait values (GroupMetric below) or on something heavier
    like pooled pixels (metrics.color_clusters).

    - `group_col` -- occurrence column naming the group. None returns an empty
      mapping, which puts every occurrence on the population-wide fallback.
    """
    if not group_col:
        return {}

    occurrence_records.require_columns(project_path, group_col,
                                       "nothing to group by")

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
    """
    `(fit, group)` for one occurrence: its own group's, or the population-wide fallback.

    Shared by every metric fit per group -- on stored trait values, or on
    pooled pixels -- so "which model scored this, and what does its score
    mean" is answered one way.

    - `fits` -- `{group: fitted thing}`, including a POPULATION entry.
    - `group_by_id` -- `{occurrence_id: group}` (see `group_lookup`).
    - `occurrence_id` -- the occurrence being scored.
    """
    group = group_by_id.get(occurrence_id, POPULATION)
    fit = fits.get(group)
    if fit is None:
        group, fit = POPULATION, fits[POPULATION]
    return fit, group


class PooledPixelGroupMetric(Metric):
    """
    Base for a metric fit per group on POOLED PIXELS rather than on stored values.

    The other half of the group-metric story. `GroupMetric` fits one ready-made
    feature row per reference occurrence, read from an earlier run's values;
    this one reads every reference occurrence's image and pools thousands of
    pixels per group -- so the fit is expensive, happens once in `prepare()`,
    and is what a colour vocabulary discovered from a population needs.

    A subclass supplies `_fit(pixels)` and `_score(segment)`, and describes
    itself in `spec()`; everything else here is the pooling, the
    minimum-group-size fallback, and the fit record.

    - `transforms` -- operations applied when gathering the reference pixels.
      Pass the SAME ones the run uses, or the fit describes a different
      representation than the one being scored.
    - `group_col` -- occurrence column to group by. None fits one population.
    - `min_group_size` -- groups with fewer reference occurrences than this
      share the population-wide fit.
    - `sample_pixels` -- pixels sampled per reference occurrence when POOLING.
      Scoring is the subclass's business and normally samples nothing.
    - `reference` -- fit against reference masks instead of canonical ones.
    """

    def __init__(self, name, score_fn, group_col=None, transforms=(),
                 min_group_size=MIN_GROUP_SIZE, sample_pixels=2000,
                 reference=False, metric_name=None, unit="category", version="1"):
        super().__init__(name, score_fn, version=version, unit=unit,
                         metric_name=metric_name or name)
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
        """
        One segment's masked pixels for pooling, or None where there's no mask.

        Subclasses that pool something other than raw BGR override this.
        """
        return masked_pixels(segment, cap=self.sample_pixels)

    def _describe_fit(self, group):
        """Extra keys about one group's fit for the run record, e.g. what it found."""
        return {}

    def prepare(self, context):
        """
        Pool every reference occurrence's masked pixels per group and fit each group.

        Expensive -- it reads every reference image -- so it happens once per
        run through this hook rather than once per occurrence.

        Returns the fit record the run stores: each fit's contributing
        occurrences as a count and a digest, with groups too small to fit
        their own marked fitted=False.
        """
        from ..segments import iterate_segments

        self.group_by_id = group_lookup(context.project_path, self.group_col,
                                        context.occurrence_ids)

        pooled = {}
        contributors = {}
        for occurrence_id, segment in iterate_segments(
                context.project_path, part=context.part, transforms=self.transforms,
                reference=self.reference, occurrence_ids=context.occurrence_ids):
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
                    group, len(chunks), self.min_group_size)
            if group is not POPULATION:
                record = dict(ids_record(contributors[group]), fitted=fitted)
                if fitted:
                    record.update(self._describe_fit(group))
                groups[str(group)] = record

        return {
            "group_col": self.group_col,
            "min_group_size": self.min_group_size,
            "reference": self.reference,
            "population": dict(ids_record(contributors[POPULATION]),
                               **self._describe_fit(POPULATION)),
            "groups": groups,
        }

    def _fit_for(self, occurrence_id):
        """This occurrence's group's fit, falling back to the population-wide one."""
        if not self.fits:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you; calling "
                "the operation directly skips it"
            )
        return fitted_for(self.fits, self.group_by_id, occurrence_id)


def _estimator_spec(model):
    """An sklearn model's configuration as JSON: its class and scalar parameters, per pipeline step."""
    steps = getattr(model, "steps", None)
    if steps is not None:
        return {"pipeline": [_estimator_spec(step) for _name, step in steps]}
    params = {}
    if hasattr(model, "get_params"):
        params = {key: value
                  for key, value in sorted(model.get_params(deep=False).items())
                  if value is None or isinstance(value, (str, int, float, bool))}
    return {"class": type(model).__name__, "params": params}


class GroupMetric(StoredValueMetric):
    """
    Base for metrics fit once per group over a population of stored values.

    The population is visible only to the fit in `prepare()`; each occurrence is
    then scored on its own stored row. Usable directly by supplying
    model_factory and score_fn; OutlierMetric and ClusterMetric below are the
    two cases that come up. A feature may be vector-valued (an embedding): each
    element becomes one column of the fit.

    - `features` -- ordered Metric operations making up the feature vector,
      e.g. [body_length(), max_width()]. Each must be an operation of
      `from_run`'s current recipe.
    - `from_run` -- the earlier metric run holding those stored values.
    - `group_col` -- occurrence column to group by before fitting, e.g.
      "taxon". None fits one population-wide model.
    - `min_group_size` -- groups smaller than this fall back to the population
      model.
    - `model_factory` -- zero-arg callable returning a fresh, unfit model with
      .fit(X), X being one row per reference occurrence.
    - `score_fn` -- callable (model, features) -> dict, features a 1xN array.
    - `default_model_factory` -- the subclass's own default model; a configured
      model differing from it reaches the hash.
    """

    def __init__(self, features, from_run, group_col=None,
                 min_group_size=MIN_GROUP_SIZE, model_factory=None,
                 score_fn=None, name=None, metric_name=None, unit="category",
                 version="1", default_model_factory=None):
        if model_factory is None or score_fn is None:
            raise ValueError(
                f"{type(self).__name__} needs both a model_factory (zero-arg, "
                "returns an unfit model with .fit(X)) and a "
                "score_fn(model, features) -> dict"
            )
        if not features:
            raise ValueError("a group metric needs at least one feature")

        super().__init__(name or type(self).__name__.lower(), self._score,
                         features, from_run, version=version, unit=unit,
                         metric_name=metric_name or name or type(self).__name__.lower())

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
        """
        Identity includes which features, which reference run, which grouping
        and how the model is configured. The FITTED MODEL isn't in the hash:
        it's determined by the reference values, which are recorded beside it.
        """
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
        """
        Fit the reference models, once, before the run's per-occurrence loop.

        Fits against every occurrence this run covers, including ones already
        scored by an earlier interrupted attempt.

        Returns the fit record the run stores: which reference run and recipe,
        which features, which grouping, and each group's reference occurrences
        as a count and a digest, with the ones too small to fit their own model
        marked fitted=False. Also writes the reference population as a figure
        through `context.report`, when the run is visualizing.
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
                        group, len(clean), self.min_group_size,
                    )
                groups[str(group)] = dict(ids_record(clean[ID_COL]), fitted=fitted)

        population = reference.dropna(subset=columns)
        if population.empty:
            raise ValueError(
                "no reference occurrences have every feature value populated"
            )
        self._fit_group(POPULATION, population)
        self._stored = dict(zip(population[ID_COL],
                                population[columns].to_numpy(dtype=float)))

        logger.info("%s fit: %d group model(s) + 1 population-wide fallback, "
                    "%d reference occurrences", self.metric_name,
                    len(self.models) - 1, len(population))

        if context.report:
            context.report.figure(f"{self.metric_name}__reference",
                                  self._reference_figure(population, columns))

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
        """Fit one group's model on its rows, keeping its own labels where it can't predict."""
        model = self._fit(rows[self.columns])
        self.models[key] = model
        if not hasattr(model, "predict") and hasattr(model, "labels_"):
            self._fit_labels[key] = dict(zip(rows[ID_COL], model.labels_))

    def _figure_labels(self, population):
        """A label per reference occurrence to colour the reference figure by, or None."""
        if not self.group_col:
            return None
        labels = population[self.group_col].astype(object).where(
            population[self.group_col].notna(), "(none)").astype(str)
        named = set(labels.value_counts().head(FIGURE_GROUPS).index)
        return labels.where(labels.isin(named), "(other)").tolist()

    def _reference_figure(self, population, columns):
        """
        The fitted population: two features against each other, one's
        histogram, or a 2-D PCA projection when a vector feature is involved.
        """
        title = (f"{self.metric_name}: reference from '{self.from_run}' "
                 f"(n={len(population)})")
        groups = self._figure_labels(population)

        if len(columns) > len(self.features) or len(columns) > 2:
            projected = _project_2d(population[columns].to_numpy(dtype=float))
            return figures.scatter(projected[:, 0], projected[:, 1], groups=groups,
                                   xlabel="PC1", ylabel="PC2", title=title)
        if len(columns) == 2:
            return figures.scatter(population[columns[0]], population[columns[1]],
                                   groups=groups,
                                   xlabel=columns[0], ylabel=columns[1], title=title)
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
        """This occurrence's group's model, falling back to the population-wide one."""
        return fitted_for(self.models, self.group_by_id, occurrence_id)

    def _score_features(self, model, group, occurrence_id, features):
        """score_fn's dict for one occurrence's feature row (a 1xN array)."""
        return self.score_fn(model, features)

    def _score(self, target):
        """
        Score one occurrence's stored row against its group's model.

        - `target` -- a `StoredValues`; run_metrics builds it.

        Returns score_fn's dict plus a "group" key naming which group actually
        scored it (None where the population-wide fallback was used).
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
    """Rows projected onto their first two principal components, zero-padded when fewer exist."""
    centered = matrix - matrix.mean(axis=0)
    if len(centered) < 2:
        return np.zeros((len(centered), 2))
    _u, _s, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((len(projected), 1))])
    return projected


def _isolation_forest_score(model, features):
    """
    is_outlier is IsolationForest's own -1/1 call (governed by
    `contamination`); anomaly_score is the continuous decision function beneath
    it, where LOWER is more anomalous. Both are stored so a stricter or looser
    cutoff can be applied later at export without refitting anything.
    """
    return {
        "is_outlier": bool(model.predict(features)[0] == -1),
        "anomaly_score": float(model.decision_function(features)[0]),
    }


def _default_isolation_forest(contamination="auto"):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(contamination=contamination, random_state=0)


class OutlierMetric(GroupMetric):
    """
    Flags occurrences that are unusual within their own group's trait
    distribution.

    Being an outlier WITHIN your own group is the QC-relevant signal; comparing
    across groups mostly rediscovers real between-group differences. Defaults
    to IsolationForest, which needs no cluster count; swap in anything else
    via model_factory.

    - `contamination` -- expected fraction of outliers in each group's
      reference population; passed straight to IsolationForest. `"auto"`
      lets it decide. Ignored if `model_factory` is given.
    - `name` -- what this measurement is called: it sets `metric_name` and
      nothing else. The operation stays `"outlier"`, which is what logs and
      `quality.WARN_THRESHOLDS` read.

    Everything else is passed through to GroupMetric -- see its docstring.
    """

    def __init__(self, features, from_run, group_col=None,
                 min_group_size=MIN_GROUP_SIZE, contamination="auto",
                 model_factory=None, name=None, unit="category"):
        model_factory = model_factory or (
            lambda: _default_isolation_forest(contamination))
        super().__init__(features, from_run, group_col=group_col,
                         min_group_size=min_group_size,
                         model_factory=model_factory,
                         score_fn=_isolation_forest_score,
                         name="outlier", metric_name=name, unit=unit,
                         default_model_factory=_default_isolation_forest)


def _cluster_score(model, features, probability_threshold=None):
    """
    Which cluster this occurrence lands in, plus how firmly where the model can
    say: distance to that cluster's centre where it has centres, the cluster's
    probability where it is probabilistic.

    - `probability_threshold` -- below this probability the cluster is -1.
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
    """`model` behind standardization and a PCA down to `n_components`."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         PCA(n_components=n_components, random_state=0), model)


def _cutout(segment):
    """The segment's masked pixels on the grid background, cropped to the mask; the image when there's no mask."""
    image = np.asarray(segment.image)
    if image.ndim == 2:
        image = np.dstack([image] * 3)
    if segment.mask is None:
        return image
    mask = np.asarray(segment.mask) > 0
    if not mask.any():
        return image
    out = np.empty_like(image)
    out[:] = grids.BACKGROUND
    out[mask] = image[mask]
    box = mask_bounds(mask)
    return out[box["y"]:box["y"] + box["height"], box["x"]:box["x"] + box["width"]]


class ClusterMetric(GroupMetric):
    """
    Which cluster an occurrence falls into within its own group's trait
    distribution, and how firmly.

    Cluster ids are only comparable within one group, since each group's model
    numbers its own. Any sklearn clusterer works through `model_factory`; one
    without `predict()` (DBSCAN, HDBSCAN) reports its own fit labels, -1 being
    noise. With the run visualizing,
    `prepare()` also writes a gallery per group: a row of sampled members per
    cluster.

    - `n_clusters` -- clusters per group, passed to the default KMeans.
      Ignored if `model_factory` is given.
    - `n_components` -- standardize and PCA-reduce the features to this many
      dimensions before clustering, e.g. for an embedding.
    - `probability_threshold` -- with a probabilistic model (e.g. a
      GaussianMixture `model_factory`), a cluster probability below this is
      reported as cluster -1.
    - `name` -- what this measurement is called; the operation stays `"cluster"`.

    Everything else is passed through to GroupMetric -- see its docstring.
    """

    def __init__(self, features, from_run, group_col=None, n_clusters=3,
                 min_group_size=MIN_GROUP_SIZE, model_factory=None, name=None,
                 unit="category", n_components=None,
                 probability_threshold=None):
        base_factory = model_factory or (lambda: _default_kmeans(n_clusters))
        factory = (base_factory if n_components is None
                   else lambda: _reduced(base_factory(), n_components))
        if (probability_threshold is not None
                and not hasattr(factory(), "predict_proba")):
            raise ValueError(
                "probability_threshold needs a model with predict_proba(), "
                "e.g. model_factory=lambda: GaussianMixture(3)")

        super().__init__(features, from_run, group_col=group_col,
                         min_group_size=min_group_size,
                         model_factory=factory,
                         score_fn=partial(_cluster_score,
                                          probability_threshold=probability_threshold),
                         name="cluster", metric_name=name, unit=unit,
                         default_model_factory=_default_kmeans)
        self.probability_threshold = probability_threshold

    def spec(self):
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
        """{group: {cluster_id: [occurrence ids]}} over the reference population."""
        assignments = {}
        for occurrence_id, row in self._stored.items():
            model, group = self._model_for(occurrence_id)
            cluster_id = self._score_features(model, group, occurrence_id,
                                              row[None, :])["cluster_id"]
            assignments.setdefault(group, {}).setdefault(cluster_id, []).append(occurrence_id)
        return assignments

    def _figure_labels(self, population):
        if self.group_col:
            return super()._figure_labels(population)
        labels = []
        for occurrence_id, row in zip(population[ID_COL],
                                      population[self.columns].to_numpy(dtype=float)):
            model, group = self._model_for(occurrence_id)
            labels.append(f"cluster {self._score_features(model, group, occurrence_id, row[None, :])['cluster_id']}")
        return labels

    def prepare(self, context):
        record = super().prepare(context)
        if context.report:
            self._draw_galleries(context)
        return record

    def _draw_galleries(self, context):
        """One grid per group, the largest FIGURE_GROUPS of them: a row of sampled members per cluster."""
        from ..segments import iterate_segments

        assignments = self._assignments()
        largest = sorted(assignments, key=lambda group: -sum(
            len(ids) for ids in assignments[group].values()))[:FIGURE_GROUPS]
        shown = {group: {cluster_id: sample_occurrences(ids, GALLERY_MEMBERS)
                         for cluster_id, ids in assignments[group].items()}
                 for group in largest}
        wanted = sorted({occurrence_id for clusters in shown.values()
                         for ids in clusters.values() for occurrence_id in ids})

        cells = {}
        for occurrence_id, segment in iterate_segments(
                context.project_path, part=context.part,
                transforms=context.transforms, reference=context.reference,
                occurrence_ids=wanted, progress=f"{self.metric_name} gallery"):
            cells[occurrence_id] = _cutout(segment)
        if not cells:
            return

        for group in largest:
            clusters = sorted(shown[group])
            rows = [[cells[occurrence_id] for occurrence_id in shown[group][cluster_id]
                     if occurrence_id in cells] for cluster_id in clusters]
            if not any(rows):
                continue
            labels = [f"{cluster_id} n={len(assignments[group][cluster_id])}"
                      for cluster_id in clusters]
            title = f"{self.metric_name}: clusters of '{self.from_run}'"
            suffix = ""
            if group is not POPULATION:
                title += f", group {group}"
                suffix = f"__{group}"
            grid = grids.comparison_grid(rows, row_labels=labels, title=title)
            context.report.figure(f"{self.metric_name}__clusters{suffix}", grid)


def outlier(features, from_run, **kwargs):
    """Operation: OutlierMetric, in the lowercase factory style of every other metric."""
    return OutlierMetric(features, from_run, **kwargs)


def cluster(features, from_run, **kwargs):
    """Operation: ClusterMetric, in the lowercase factory style of every other metric."""
    return ClusterMetric(features, from_run, **kwargs)
