import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.metrics import matthews_corrcoef
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique
from sklearn import preprocessing
from sklearn.utils.random import check_random_state

from resilient.logger import Logger

from resilient.selection_optimizers import GridOptimizer
from resilient.selection_strategies import SelectBestPercent
from resilient.train_set_generators import CentroidBasedPDFTrainSetGenerator
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

MAX_INT = np.iinfo(np.int32).max


class TrainingStrategy(BaseEstimator):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_features='auto'),
                 train_set_generator=CentroidBasedPDFTrainSetGenerator()):
        self.base_estimator = base_estimator
        self.train_set_generator = train_set_generator

    def train_estimators(self, inp, y, weighting_strategy, random_state):
        classifiers = []
        for i, sample_weights in enumerate(self.train_set_generator.get_sample_weights(inp, y, random_state)):
            Logger.get().write("!Training estimator:", (i+1))
            est = self._make_estimator(inp, y, sample_weights, random_state)
            weighting_strategy.add_estimator(est, train_set)
            classifiers.append(est)
        return classifiers

    def _make_estimator(self, train_set, random_state):
        seed = random_state.randint(MAX_INT)
        est = clone(self.base_estimator)
        est.set_params(random_state=check_random_state(seed))
        est.fit(train_set.data, train_set.target)
        return est


class ResilientEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 training_strategy=TrainingStrategy(),
                 weighting_strategy=CentroidBasedWeightingStrategy(),
                 selection_strategy=SelectBestPercent(),
                 selection_optimizer=GridOptimizer(),
                 validation_percent=0.1,
                 multiply_by_weight=False,
                 use_prob=True,
                 random_state=None):
        self.training_strategy = training_strategy
        self.weighting_strategy = weighting_strategy
        self.selection_strategy = selection_strategy
        self.selection_optimizer = selection_optimizer
        self.validation_percent = validation_percent
        self.multiply_by_weight = multiply_by_weight
        self.use_prob = use_prob
        self.random_state = random_state
        # Training time attributes
        self.classes_ = None
        self.n_classes_ = None
        self.classifiers_ = None
        self.precomputed_probs_ = None
        self.precomputed_weights_ = None
        self.random_state_ = None

    def fit(self, inp, y):
        self.precomputed_probs_ = None
        self.precomputed_weights_ = None

        self.classes_, y = unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.random_state_ = check_random_state(self.random_state)

        n_samples = len(inp)

        if self.validation_percent is not None:
            train_indices = self.random_state_.choice(n_samples, size=int(n_samples*(1.0-self.validation_percent)),
                                                      replace=False)
            train_inp, train_y = inp[train_indices], y[train_indices]
        else:
            train_indices = np.ones(n_samples, dtype=bool)
            train_inp, train_y = inp, y

        self.weighting_strategy.prepare(train_inp, train_y)
        self.classifiers_ = self.training_strategy.train_estimators(train_inp, train_y,
                                                                    self.weighting_strategy, self.random_state_)

        if self.validation_percent is not None:
            if self.validation_percent > 0.0:
                valid_mask = np.ones(n_samples, dtype=bool)
                valid_mask[train_indices] = False
            else:
                valid_mask = train_indices
            valid_inp, valid_y = inp[valid_mask], y[valid_mask]
            self.selection_optimizer.optimize(self, valid_inp, valid_y)

        # Reset it to null because the previous line uses self.predict
        self.precomputed_probs_ = None
        self.precomputed_weights_ = None
        return self

    def predict_proba(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N, n_classes_), each row sums to one
        if self.precomputed_probs_ is None:
            self._precompute(inp)
        prob = np.zeros((len(inp), self.n_classes_))
        for i in range(len(inp)):
            active_indices = self.selection_strategy.get_indices(self.precomputed_weights_[i], self.random_state_)
            prob[i] = self.precomputed_probs_[i][active_indices].sum(axis=0)
        preprocessing.normalize(prob, norm='l1', copy=False)
        return prob

    def predict(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N), one label per instance
        p = self.predict_proba(inp)
        return self.classes_[np.argmax(p, axis=1)]

    def _precompute(self, inp):
        self.precomputed_probs_ = np.zeros((len(inp), len(self.classifiers_), self.n_classes_))
        self.precomputed_weights_ = np.zeros((len(inp), len(self.classifiers_)))
        for i, x in enumerate(inp):
            for j, cls in enumerate(self.classifiers_):
                prob = cls.predict_proba(x)[0]
                if not self.use_prob:
                    max_index = prob.argmax()
                    prob = np.zeros_like(prob)
                    prob[max_index] = 1
                self.precomputed_probs_[i][j] = prob
            self.precomputed_weights_[i] = self.weighting_strategy.weight_estimators(x)
            if self.multiply_by_weight:
                for j in range(len(self.classifiers_)):
                    self.precomputed_probs_[i][j] *= self.precomputed_weights_[i][j]
            Logger.get().write("!Computing", len(inp), "probabilities and weights:", (i+1))

    def score(self, inp, y, use_mcc=False):
        if use_mcc:
            return matthews_corrcoef(y, self.predict(inp))
        else:
            return super(ResilientEnsemble, self).score(inp, y)
