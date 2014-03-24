import hashlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique
from sklearn import preprocessing
from sklearn.utils.random import check_random_state

from resilient.logger import Logger
from resilient.selection_strategies import SelectBestPercent
from resilient.train_set_generators import RandomCentroidPDFTrainSetGenerator
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

MAX_INT = np.iinfo(np.int32).max


class TrainingStrategy(BaseEstimator):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_features='auto'),
                 train_set_generator=RandomCentroidPDFTrainSetGenerator()):
        self.base_estimator = base_estimator
        self.train_set_generator = train_set_generator

    def train_estimators(self, n, inp, y, weighting_strategy, random_state):
        classifiers = []
        for i, sample_weights in enumerate(self.train_set_generator.get_sample_weights(n, inp, y, random_state)):
            Logger.get().write("!Training estimator:", (i+1))
            est = self._make_estimator(inp, y, sample_weights, random_state)
            weighting_strategy.add_estimator(est, inp, y, sample_weights)
            classifiers.append(est)
        return classifiers

    def _make_estimator(self, inp, y, sample_weights, random_state):
        seed = random_state.randint(MAX_INT)
        est = clone(self.base_estimator)
        est.set_params(random_state=check_random_state(seed))
        est.fit(inp, y, sample_weight=sample_weights)
        return est


class ResilientEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 preprocessing_pipeline=None,
                 n_estimators=10,
                 training_strategy=TrainingStrategy(),
                 weighting_strategy=CentroidBasedWeightingStrategy(),
                 selection_strategy=SelectBestPercent(),
                 multiply_by_weight=False,
                 use_prob=True,
                 random_state=None):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.n_estimators = n_estimators
        self.training_strategy = training_strategy
        self.weighting_strategy = weighting_strategy
        self.selection_strategy = selection_strategy
        self.multiply_by_weight = multiply_by_weight
        self.use_prob = use_prob
        self.random_state = random_state
        # Training time attributes
        self.classes_ = None
        self.n_classes_ = None
        self.classifiers_ = None
        self.precomputed_probs_ = None
        self.precomputed_weights_ = None
        self.pipeline_ = None
        self.random_state_ = None

    def fit(self, inp, y):
        self.precomputed_probs_ = None
        self.precomputed_weights_ = None

        self.classes_, y = unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.random_state_ = check_random_state(self.random_state)

        if self.preprocessing_pipeline is not None:
            self.pipeline_ = self.preprocessing_pipeline.fit(inp)
            inp = self.pipeline_.transform(inp)

        self.weighting_strategy.prepare(inp, y)
        self.classifiers_ = self.training_strategy.train_estimators(
            self.n_estimators, inp, y, self.weighting_strategy, self.random_state_
        )

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
        # output is array-like, N, one label per instance
        if self.pipeline_ is not None:
            inp = self.pipeline_.transform(inp)
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

    def get_directory(self):
        custom_state = self.random_state
        custom_selection = self.selection_strategy
        self.random_state = None
        self.selection_strategy = None
        filename = hashlib.md5(repr(self)).hexdigest()
        self.random_state = custom_state
        self.selection_strategy = custom_selection
        return filename

    def get_filename(self):
        return self.get_directory() + "/ensemble"
