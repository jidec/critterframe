"""
Group metrics: metrics that need a reference population before any single
occurrence can be scored.

Every other metric in this package is self-contained -- hand it one segment and
it returns a value. "Is this specimen an outlier?" isn't answerable that way.
It needs to know what the rest of the population looks like first, and usually
what the rest of THIS SPECIES looks like, since a body length that's
unremarkable for one species is impossible for another.

A group metric is still a metric. It stores like one, exports like one, and
composes into a recipe like one. The only difference is that it fits once
before the run's per-occurrence loop -- via the prepare() hook every operation
has and almost none use -- and then scores each occurrence against whichever
group's fitted model applies.

Where the reference values come from: a group metric doesn't recompute the
population's traits, it READS them from an earlier metric run (`from_run`).
That earlier run already measured every occurrence and stored the result, so
refitting from stored values is both cheaper and more honest -- the population
you're scoring against is exactly the one recorded, not a re-derivation that
might differ if a transform changed in between. The occurrence being SCORED is
measured fresh from its own segment, so an occurrence that wasn't in the
reference population still gets a value.

Two concrete cases, differing only in what model is fit and how it's read:

  OutlierMetric -- is this occurrence unusual within its group
                   (IsolationForest by default).
  ClusterMetric -- which cluster within its group does it fall into, and how
                   far from that cluster's centre (KMeans by default).

They answer different questions. A point can be a perfectly unremarkable member
of no particular sub-cluster, or an outlier despite landing nearest to one.
"""

import logging

import numpy as np

from ..project import paths
from ..recipes import Metric
from ..records.metrics import latest_values
from ..records.occurrences import ID_COL, load_occurrences
from ..storage.tables import table_columns

logger = logging.getLogger(__name__)

# Key for the population-wide fallback model. Every occurrence whose group is
# unknown, or whose group is too small to fit its own model, is scored against
# this one rather than going unscored -- a missing species label should cost you
# precision, not the measurement.
POPULATION = None

# Groups with fewer reference occurrences than this don't get their own model.
# Fitting an outlier detector on three points produces confident nonsense.
MIN_GROUP_SIZE = 5


def group_lookup(project_path, group_col, occurrence_ids=None):
    """
    {occurrence_id: group} read off the occurrence table.

    Shared rather than reimplemented per group metric: "which group does this
    occurrence belong to, with a population-wide fallback when the answer is
    missing" is the same question for every metric fit per group, whether it
    fits on stored trait values (GroupMetric below) or on something heavier
    like pooled pixels (extensions.inat_insects.metrics.color).

    group_col -- occurrence column naming the group. None returns an empty
                 mapping, which puts every occurrence on the population-wide
                 fallback.
    """
    if not group_col:
        return {}

    # Checked against the parquet's schema before reading, the same way a
    # calibration scope is: naming a column the table doesn't have fails inside
    # pyarrow with a message about FieldRefs that says nothing about groups.
    available = table_columns(paths.occurrences_path(project_path))
    if group_col not in available:
        raise KeyError(
            f"no '{group_col}' column in this project's occurrences, so there "
            f"is nothing to group by (columns: {sorted(available)})"
        )

    groups = load_occurrences(project_path, columns=[group_col])
    if occurrence_ids is not None:
        groups = groups[groups[ID_COL].isin({str(i) for i in occurrence_ids})]
    return groups.set_index(ID_COL)[group_col].to_dict()


class GroupMetric(Metric):
    """
    Base for metrics fit once per group over a reference population.

    Usable directly -- supply model_factory and score_fn for a one-off group
    metric that doesn't warrant its own class -- but OutlierMetric and
    ClusterMetric below are the two cases that come up.

    features       -- ordered list of Metric operations whose values make up
                      the feature vector, e.g. [body_length(), max_width()].
                      The same operations are used both to look up the
                      reference population's stored values and to measure the
                      occurrence being scored, so the two can't drift apart.
    from_run       -- name of the earlier metric run holding those stored
                      values. Its part is taken from the run this group metric
                      is running in.
    group_col      -- occurrence-table column to group by before fitting, e.g.
                      "taxon" for per-species models. None fits a single
                      population-wide model for everyone.
    min_group_size -- groups smaller than this fall back to the population-wide
                      model.
    model_factory  -- zero-arg callable returning a fresh, unfit model with a
                      .fit(X) method, X being a 2D array with one row per
                      reference occurrence.
    score_fn       -- callable (model, features) -> dict, where features is a
                      2D array holding exactly one row. Separate from
                      model_factory because model types don't share a scoring
                      API -- KMeans is read with .transform(), IsolationForest
                      with .decision_function() -- and that difference belongs
                      in one small function rather than in this class.
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

        if self.group_col:
            self.group_by_id = reference.set_index(ID_COL)[self.group_col].to_dict()
            for group, rows in reference.groupby(self.group_col):
                clean = rows[columns].dropna()
                if len(clean) < self.min_group_size:
                    logger.warning(
                        "group %r has only %d reference occurrences (< "
                        "min_group_size=%d) -- scoring it against the "
                        "population-wide model instead of its own",
                        group, len(clean), self.min_group_size,
                    )
                    continue
                self.models[group] = self._fit(clean)

        population = reference[columns].dropna()
        if population.empty:
            raise ValueError(
                "no reference occurrences have every feature value populated"
            )
        self.models[POPULATION] = self._fit(population)

        logger.info("%s fit: %d group model(s) + 1 population-wide fallback, "
                    "%d reference occurrences", self.metric_name,
                    len(self.models) - 1, len(population))

    def _reference_table(self, context):
        """
        Assemble the reference population: one row per occurrence, one column
        per feature, plus the group column if there is one.
        """
        import pandas as pd

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
        group = self.group_by_id.get(occurrence_id, POPULATION)
        model = self.models.get(group)
        if model is None:
            group, model = POPULATION, self.models[POPULATION]
        return model, group

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

    contamination -- expected fraction of outliers in each group's reference
                     population; passed straight to IsolationForest. "auto"
                     lets it decide. Ignored if model_factory is given.

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
                         name=name or "outlier", unit=unit)


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

    n_clusters -- clusters per group, passed to the default KMeans
                  model_factory. Ignored if model_factory is given.

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
                         name=name or "cluster", unit=unit)


def outlier(features, from_run, **kwargs):
    """Operation: OutlierMetric, in the lowercase factory style of every other metric."""
    return OutlierMetric(features, from_run, **kwargs)


def cluster(features, from_run, **kwargs):
    """Operation: ClusterMetric, in the lowercase factory style of every other metric."""
    return ClusterMetric(features, from_run, **kwargs)
