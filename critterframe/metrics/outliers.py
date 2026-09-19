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

import numpy as np
import pandas as pd

from ..recipes import Metric
from .pixels import masked_pixels
from ..records import occurrences as occurrence_records
from ..records.metrics import latest_values
from ..records.occurrences import ID_COL, ids_record, load_occurrences
from ..visualization import figures

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


class GroupMetric(Metric):
    """
    Base for metrics fit once per group over a reference population.

    Usable directly by supplying model_factory and score_fn; OutlierMetric and
    ClusterMetric below are the two cases that come up.

    - `features` -- ordered Metric operations making up the feature vector,
      e.g. [body_length(), max_width()]. The same operations look up the
      reference values and measure the occurrence being scored, so the two
      can't drift apart.
    - `from_run` -- the earlier metric run holding those stored values.
    - `group_col` -- occurrence column to group by before fitting, e.g.
      "taxon". None fits one population-wide model.
    - `min_group_size` -- groups smaller than this fall back to the population
      model.
    - `model_factory` -- zero-arg callable returning a fresh, unfit model with
      .fit(X), X being one row per reference occurrence.
    - `score_fn` -- callable (model, features) -> dict. Separate from
      model_factory because model types don't share a scoring API -- KMeans is
      read with .transform(), IsolationForest with .decision_function().
    """

    def __init__(self, features, from_run, group_col=None,
                 min_group_size=MIN_GROUP_SIZE, model_factory=None,
                 score_fn=None, name=None, metric_name=None, unit="category",
                 version="1"):
        if model_factory is None or score_fn is None:
            raise ValueError(
                f"{type(self).__name__} needs both a model_factory (zero-arg, "
                "returns an unfit model with .fit(X)) and a "
                "score_fn(model, features) -> dict"
            )
        if not features:
            raise ValueError("a group metric needs at least one feature")

        super().__init__(name or type(self).__name__.lower(), self._score,
                         version=version, unit=unit,
                         metric_name=metric_name or name or type(self).__name__.lower())

        self.features = list(features)
        self.from_run = from_run
        self.group_col = group_col
        self.min_group_size = min_group_size
        self.model_factory = model_factory
        self.score_fn = score_fn

        self.models = {}
        self.group_by_id = {}

    def spec(self):
        """
        Identity includes which features, which reference run, and which
        grouping -- all three change what "outlier" means, so all three have to
        change the hash. The FITTED MODEL isn't in the hash: it's determined by
        the reference values, which are determined by from_run, so hashing it
        would add nothing but instability from model randomness.
        """
        spec = super().spec()
        spec["parameters"] = {
            "features": [feature.spec() for feature in self.features],
            "from_run": self.from_run,
            "group_col": self.group_col,
            "min_group_size": self.min_group_size,
            "model": type(self.model_factory()).__name__,
        }
        return spec

    def prepare(self, context):
        """
        Fit the reference models, once, before the run's per-occurrence loop.

        Fits against every occurrence this run covers, including ones already
        scored by an earlier interrupted attempt -- the reference population has
        to be the whole population being scored, or resuming a run would
        silently change what the score means partway through.

        Returns the fit record the run stores: which reference run, which
        features, which grouping, and each group's reference occurrences as a
        count and a digest, with the ones too small to fit their own model
        marked fitted=False. Also writes the reference population as a figure
        through `context.report`, when the run is visualizing.
        """
        reference = self._reference_table(context)
        if reference.empty:
            raise ValueError(
                f"no reference values for {self.metric_name}: run "
                f"'{self.from_run}' has no stored "
                f"{[feature.metric_name for feature in self.features]} values "
                f"for part '{context.part}'. Run those metrics first -- a group "
                "metric scores against a population that has already been "
                "measured."
            )

        columns = [feature.metric_name for feature in self.features]

        groups = {}
        if self.group_col:
            self.group_by_id = reference.set_index(ID_COL)[self.group_col].to_dict()
            for group, rows in reference.groupby(self.group_col):
                # Dropped by subset rather than by projection so the ids stay
                # beside the numbers: the same rows fit the model and name it.
                clean = rows.dropna(subset=columns)
                fitted = len(clean) >= self.min_group_size
                if fitted:
                    self.models[group] = self._fit(clean[columns])
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
        self.models[POPULATION] = self._fit(population[columns])

        logger.info("%s fit: %d group model(s) + 1 population-wide fallback, "
                    "%d reference occurrences", self.metric_name,
                    len(self.models) - 1, len(population))

        if context.report:
            context.report.figure(f"{self.metric_name}__reference",
                                  self._reference_figure(population, columns))

        return {
            "from_run": self.from_run,
            "group_col": self.group_col,
            "features": columns,
            "model": type(self.model_factory()).__name__,
            "min_group_size": self.min_group_size,
            "population": ids_record(population[ID_COL]),
            "groups": groups,
        }

    def _reference_figure(self, population, columns):
        """The fitted population: the first two features against each other, or one's histogram."""
        title = (f"{self.metric_name}: reference from '{self.from_run}' "
                 f"(n={len(population)})")
        groups = None
        if self.group_col:
            labels = population[self.group_col].astype(object).where(
                population[self.group_col].notna(), "(none)").astype(str)
            named = set(labels.value_counts().head(FIGURE_GROUPS).index)
            groups = labels.where(labels.isin(named), "(other)")

        if len(columns) >= 2:
            return figures.scatter(population[columns[0]], population[columns[1]],
                                   groups=None if groups is None else groups.tolist(),
                                   xlabel=columns[0], ylabel=columns[1], title=title)
        values = (population[columns[0]].tolist() if groups is None else
                  {group: rows.tolist() for group, rows in population[columns[0]].groupby(groups)})
        return figures.histogram(values, xlabel=columns[0], title=title)

    def _reference_table(self, context):
        """
        Assemble the reference population: one row per occurrence, one column
        per feature, plus the group column if there is one.
        """
        columns = {}
        for feature in self.features:
            columns[feature.metric_name] = latest_values(
                context.project_path, self.from_run, part=context.part,
                metric_name=feature.metric_name,
            )

        reference = pd.DataFrame(columns)
        if reference.empty:
            return reference

        reference.index.name = ID_COL
        reference = reference.reset_index()
        reference = reference[reference[ID_COL].isin(set(context.occurrence_ids))]

        for column in [feature.metric_name for feature in self.features]:
            reference[column] = pd.to_numeric(reference[column], errors="coerce")

        if self.group_col:
            groups = load_occurrences(context.project_path,
                                      columns=[self.group_col])
            reference = reference.merge(groups, on=ID_COL, how="left")

        return reference

    def _fit(self, frame):
        model = self.model_factory()
        model.fit(frame.to_numpy())
        return model

    def _model_for(self, occurrence_id):
        """This occurrence's group's model, falling back to the population-wide one."""
        return fitted_for(self.models, self.group_by_id, occurrence_id)

    def _score(self, segment):
        """
        Score one occurrence against its group's model.

        The feature vector is recomputed from this segment rather than looked up
        -- so an occurrence that was never in the reference population still
        gets a value, and so the features being scored come from the same
        transforms this run applied.

        Returns score_fn's dict plus a "group" key naming which group actually
        scored it (None where the population-wide fallback was used), because a
        score is not interpretable without knowing what it was scored against.
        """
        if not self.models:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- group metrics are fit by "
                "their prepare() hook, which run_metrics calls for you; calling "
                "the operation directly skips it"
            )

        model, group = self._model_for(segment.occurrence_id)
        features = [[float(feature(segment)) for feature in self.features]]

        result = dict(self.score_fn(model, np.asarray(features)))
        result["group"] = group
        return result


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


class OutlierMetric(GroupMetric):
    """
    Flags occurrences that are unusual within their own group's trait
    distribution.

    Being an outlier WITHIN your own group is the QC-relevant signal. Comparing
    across groups would mostly rediscover real between-group differences -- that
    one species is larger than another -- rather than catching the bad
    segmentations and mis-identifications this is for.

    Defaults to IsolationForest, which is built for exactly this: no cluster
    count to choose, and it isn't looking for structure in the data, just
    isolating points that are easy to separate from the rest. Swap in anything
    else (a one-class SVM, a per-group Mahalanobis distance) via
    model_factory/score_fn -- nothing else about this class is
    IsolationForest-specific.

    - `contamination` -- expected fraction of outliers in each group's
      reference population; passed straight to IsolationForest. `"auto"`
      lets it decide. Ignored if `model_factory` is given.
    - `name` -- what this measurement is CALLED, like every other metric
      factory's `name=`: it sets `metric_name` and nothing else. The
      operation stays `"outlier"` -- the operation name is what says what
      RAN, and it's what logs, error messages and `quality.WARN_THRESHOLDS`
      read.

    Everything else is passed through to GroupMetric -- see its docstring.
    """

    def __init__(self, features, from_run, group_col=None,
                 min_group_size=MIN_GROUP_SIZE, contamination="auto",
                 model_factory=None, name=None, unit="category"):
        from sklearn.ensemble import IsolationForest

        model_factory = model_factory or (
            lambda: IsolationForest(contamination=contamination, random_state=0)
        )
        super().__init__(features, from_run, group_col=group_col,
                         min_group_size=min_group_size,
                         model_factory=model_factory,
                         score_fn=_isolation_forest_score,
                         name="outlier", metric_name=name, unit=unit)


def _kmeans_score(model, features):
    """
    Which cluster this occurrence lands in, and how far from that cluster's
    centre. The distance is the useful diagnostic: large despite being the
    NEAREST centroid still means this occurrence sits far from where its peers
    cluster.
    """
    cluster_id = int(model.predict(features)[0])
    return {
        "cluster_id": cluster_id,
        "centroid_distance": float(model.transform(features)[0][cluster_id]),
    }


class ClusterMetric(GroupMetric):
    """
    Which cluster an occurrence falls into within its own group's trait
    distribution, and how far from that cluster's centre.

    Same within-group reasoning as OutlierMetric: clustering across groups would
    mostly rediscover between-group differences. Cluster ids are only comparable
    within one group, since each group's model numbers its own clusters.

    - `n_clusters` -- clusters per group, passed to the default KMeans
      `model_factory`. Ignored if `model_factory` is given.
    - `name` -- what this measurement is called; see `OutlierMetric`. It
      sets `metric_name` alone, leaving the operation `"cluster"`.

    Everything else is passed through to GroupMetric -- see its docstring.
    """

    def __init__(self, features, from_run, group_col=None, n_clusters=3,
                 min_group_size=MIN_GROUP_SIZE, model_factory=None, name=None,
                 unit="category"):
        from sklearn.cluster import KMeans

        model_factory = model_factory or (
            lambda: KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        )
        super().__init__(features, from_run, group_col=group_col,
                         min_group_size=min_group_size,
                         model_factory=model_factory, score_fn=_kmeans_score,
                         name="cluster", metric_name=name, unit=unit)


def outlier(features, from_run, **kwargs):
    """Operation: OutlierMetric, in the lowercase factory style of every other metric."""
    return OutlierMetric(features, from_run, **kwargs)


def cluster(features, from_run, **kwargs):
    """Operation: ClusterMetric, in the lowercase factory style of every other metric."""
    return ClusterMetric(features, from_run, **kwargs)
