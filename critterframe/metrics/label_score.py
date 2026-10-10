"""label_score(): the probability a person would call an occurrence-part bad, fitted on stored features and labels."""

import logging
import math

from ..selection.subsets import select_ids
from ..core.recipes import hash_spec
from ..records.metrics import latest_values
from ..records.occurrences import ID_COL, ids_record
from ..visualization import figures
from .base.stored import StoredValueMetric

logger = logging.getLogger(__name__)

# Fewer labels than this in either class and there is nothing to cross-validate.
MIN_PER_CLASS = 3

# Below this many in either class the fit runs, and the log says how thin it is.
WARN_PER_CLASS = 20

# Fixed, so one set of labels always gives one set of scores.
SEED = 0


def _classifier(n_components):
    """Return a pipeline: standardize, optionally reduce by PCA, then class-balanced logistic regression."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [StandardScaler()]
    if n_components is not None:
        steps.append(PCA(n_components=n_components, random_state=SEED))
    steps.append(LogisticRegression(class_weight="balanced", max_iter=1000))
    return make_pipeline(*steps)


class LabelScoreMetric(StoredValueMetric):
    """Operation: the fitted probability that an occurrence-part's label is a bad one.

    A labelled occurrence gets its out-of-fold prediction; every other gets the prediction
    of the model fitted on all the labels. The value is `{"bad_probability", "in_training"}`.

    Args:
        features: Metric operations whose stored values are the model's inputs, e.g.
            `[embedding(...)]`. Each must be an operation of `from_run`'s current recipe.
        from_run: The metric run holding those values.
        labels_run: The metric run holding the human labels.
        labels_subset: Named subset whose labels train the model. Required; None trains on
            every label.
        label_metric: Metric holding the labels.
        bad_labels: Label values the model predicts. Give this or `good_labels`.
        good_labels: Label values that are fine, every other being bad.
        n_components: PCA dimensions the features are reduced to before the fit; None
            skips the reduction.
        folds: Cross-validation folds for the labelled occurrences' scores.
        name: Metric name; `"label_score"` if None.
        unit: Recorded unit.
    """

    def __init__(
        self,
        features,
        from_run,
        labels_run,
        labels_subset,
        label_metric,
        bad_labels=None,
        good_labels=None,
        n_components=16,
        folds=5,
        name=None,
        unit="probability",
    ):
        if folds < 2:
            raise ValueError(f"label_score needs at least 2 folds, got {folds}")
        if (bad_labels is None) == (good_labels is None):
            raise ValueError(
                "label_score needs exactly one of bad_labels= and good_labels=, to "
                "say which labels the score predicts"
            )
        super().__init__(
            "label_score",
            self._score,
            features,
            from_run,
            version="1",
            unit=unit,
            metric_name=name or "label_score",
        )
        self.labels_run = labels_run
        self.labels_subset = labels_subset
        self.label_metric = label_metric
        self.bad_labels = None if bad_labels is None else sorted(bad_labels)
        self.good_labels = None if good_labels is None else sorted(good_labels)
        self.n_components = n_components
        self.folds = folds
        self.classifier = None
        self._scores = {}

    def spec(self):
        """Return the operation's spec, with the features, runs, label vocabulary and model settings."""
        spec = super().spec()
        spec["parameters"] = {
            "features": [feature.spec() for feature in self.features],
            "from_run": self.from_run,
            "labels_run": self.labels_run,
            "labels_subset": self.labels_subset,
            "label_metric": self.label_metric,
            # The keys keep the arguments' earlier names: they are in the recipe
            # hash, so renaming them would make every stored score stale.
            **(
                {"bad_flags": self.bad_labels}
                if self.bad_labels is not None
                else {"good_flags": self.good_labels}
            ),
            "n_components": self.n_components,
            "folds": self.folds,
        }
        return spec

    def _labels(self, context):
        """Return the training subset's current labels, as a Series of is-bad indexed by occurrence id."""
        labels = latest_values(
            context.project_path, self.labels_run, part=context.part, metric_name=self.label_metric
        ).dropna()
        if self.labels_subset is not None:
            members = set(select_ids(context.project_path, subset=self.labels_subset))
            labels = labels[labels.index.isin(members)]
        if self.bad_labels is not None:
            return labels.isin(self.bad_labels)
        return ~labels.isin(self.good_labels)

    def prepare(self, context):
        """Fit the model on the training labels and score every occurrence the run covers.

        Returns:
            The fit record: the feature run and its recipe, the labels' run and subset, the
            labelled occurrences as a count and digest, the class counts, the cross-validated
            AUC, and `fit_hash`, a digest of which occurrence carried which label.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        from_recipe_hash = self.check_from_run(context)
        is_bad = self._labels(context)

        wanted = sorted(set(context.occurrence_ids) | set(is_bad.index))
        table, columns = self.feature_table(context, occurrence_ids=wanted)
        table = table.dropna(subset=columns).set_index(ID_COL)

        train_ids = sorted(occurrence_id for occurrence_id in is_bad.index if occurrence_id in table.index)
        target = is_bad.loc[train_ids].to_numpy(dtype=bool)
        n_bad, n_good = int(target.sum()), int((~target).sum())
        if min(n_bad, n_good) < MIN_PER_CLASS:
            where = f"subset '{self.labels_subset}'" if self.labels_subset else "the project"
            raise ValueError(
                f"{self.metric_name}: {n_bad} bad and {n_good} good '{self.label_metric}' "
                f"label(s) with '{self.from_run}' features in {where} for part "
                f"'{context.part}' -- at least {MIN_PER_CLASS} of each are needed. "
                f"Label more under run '{self.labels_run}', and measure "
                f"'{self.from_run}' over the labelled occurrences."
            )
        if min(n_bad, n_good) < WARN_PER_CLASS:
            logger.warning(
                "%s: fitting on %d bad and %d good label(s) -- few "
                "enough that the score will move as labels are added",
                self.metric_name,
                n_bad,
                n_good,
            )

        folds = min(self.folds, n_bad, n_good)
        # PCA can't keep more components than the smallest fold has rows.
        n_components = self.n_components
        if n_components is not None:
            smallest_fit = len(train_ids) - math.ceil(len(train_ids) / folds)
            n_components = max(1, min(n_components, len(columns), smallest_fit))

        features = table.loc[train_ids, columns].to_numpy(dtype=float)
        out_of_fold = cross_val_predict(
            _classifier(n_components),
            features,
            target,
            cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
            method="predict_proba",
        )[:, 1]
        self.classifier = model = _classifier(n_components).fit(features, target)

        self._scores = {
            occurrence_id: (float(probability), True)
            for occurrence_id, probability in zip(train_ids, out_of_fold)
        }
        others = [occurrence_id for occurrence_id in table.index if occurrence_id not in self._scores]
        if others:
            predicted = model.predict_proba(table.loc[others, columns].to_numpy(dtype=float))[:, 1]
            self._scores.update(
                {
                    occurrence_id: (float(probability), False)
                    for occurrence_id, probability in zip(others, predicted)
                }
            )

        auc = float(roc_auc_score(target, out_of_fold))
        logger.info(
            "%s fit: %d bad, %d good label(s), %d-fold AUC %.3f, %d feature column(s)%s",
            self.metric_name,
            n_bad,
            n_good,
            folds,
            auc,
            len(columns),
            "" if n_components is None else f" reduced to {n_components}",
        )

        if context.report:
            context.report.figure(
                f"{self.metric_name}__fit",
                figures.histogram(
                    {"good": out_of_fold[~target].tolist(), "bad": out_of_fold[target].tolist()},
                    bins=20,
                    xlabel="out-of-fold bad probability",
                    title=f"{self.metric_name}: {n_bad} bad, {n_good} good, AUC {auc:.3f}",
                ),
            )

        return {
            "from_run": self.from_run,
            "from_recipe_hash": from_recipe_hash,
            "labels_run": self.labels_run,
            "labels_subset": self.labels_subset,
            "label_metric": self.label_metric,
            "features": [feature.metric_name for feature in self.features],
            "population": ids_record(train_ids),
            "n_bad": n_bad,
            "n_good": n_good,
            "folds": folds,
            "n_components": n_components,
            "cv_auc": round(auc, 4),
            "fit_hash": hash_spec(
                {"labels": [[occurrence_id, bool(bad)] for occurrence_id, bad in zip(train_ids, target)]}
            ),
        }

    def _score(self, target):
        """Return this occurrence's fitted probability, raising `NoInput` where it has no feature value."""
        occurrence_id = self.require_stored(target)
        if self.classifier is None:
            raise RuntimeError(
                f"{self.metric_name} was never fit -- its prepare() hook does "
                "that, which run_metrics calls for you"
            )
        score = self._scores.get(occurrence_id)
        if score is None:
            raise self.no_value()
        probability, in_training = score
        return {"bad_probability": probability, "in_training": in_training}


def label_score(features, from_run, labels_run, labels_subset, label_metric, **kwargs):
    """Operation: a `LabelScoreMetric`.

    Args:
        features: As in `LabelScoreMetric`.
        from_run: As in `LabelScoreMetric`.
        labels_run: As in `LabelScoreMetric`.
        labels_subset: As in `LabelScoreMetric`.
        label_metric: As in `LabelScoreMetric`.
        **kwargs: The other `LabelScoreMetric` arguments.
    """
    return LabelScoreMetric(features, from_run, labels_run, labels_subset, label_metric, **kwargs)
