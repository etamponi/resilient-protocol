import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique
from sklearn import preprocessing
from sklearn.utils.random import check_random_state

from resilient.selection_strategies import SelectBestK
from resilient.sorting_strategies import LocalScoreWeightingStrategy
from resilient.splitting_strategies import RandomChoiceSplittingStrategy
from resilient.wrapper import EstimatorWrapper


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

MAX_INT = np.iinfo(np.int32).max


class TrainingStrategy(object):

    def __init__(self,
                 n_estimators=100,
                 base_estimator=DecisionTreeClassifier(max_features='auto'),
                 splitting_strategy=RandomChoiceSplittingStrategy()):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.splitting_strategy = splitting_strategy

    def train_estimators(self, inp, y, random_state):
        classifiers = []
        i = 0
        for l_set, t_set in self.splitting_strategy.iterate(self.n_estimators, inp, y, random_state):
            i += 1
            print "\rTraining", self.n_estimators, "estimators:", i,
            est = self._make_estimator(random_state, l_set, t_set)
            classifiers.append(est)
        print ""
        return classifiers

    def _make_estimator(self, random_state, train_set, test_set):
        seed = random_state.randint(MAX_INT)
        est = clone(self.base_estimator)
        est.set_params(random_state=check_random_state(seed))
        est.fit(train_set.data, train_set.target)
        est = EstimatorWrapper(est, train_set, test_set)
        return est


class ResilientEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 training_strategy=TrainingStrategy(),
                 weighting_strategy=LocalScoreWeightingStrategy(),
                 selection_strategy=SelectBestK(),
                 multiply_by_weight=False,
                 use_prob=True,
                 random_state=None):
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
        self.random_state_ = None
        self.precomputed_prob_ = None
        self.precomputed_weights_ = None

    def fit(self, inp, y):
        self.classes_, y = unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.random_state_ = check_random_state(self.random_state)
        self.classifiers_ = self.training_strategy.train_estimators(inp, y, self.random_state_)
        self.weighting_strategy.prepare(self.classifiers_, self.random_state_)
        self.precomputed_prob_ = None
        self.precomputed_weights_ = None
        return self

    def predict_proba(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N, n_classes_), each row sums to one
        if self.precomputed_prob_ is None:
            self._precompute(inp)
        prob = np.zeros((len(inp), self.n_classes_))
        for i in range(len(inp)):
            active_indices = self.selection_strategy.select(self.precomputed_weights_[i])
            prob[i] = self.precomputed_prob_[i][active_indices].sum(axis=0)
        preprocessing.normalize(prob, norm='l1', copy=False)
        return prob

    def predict(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N), one label per instance
        p = self.predict_proba(inp)
        return self.classes_[np.argmax(p, axis=1)]

    def _precompute(self, inp):
        self.precomputed_prob_ = np.zeros((len(inp), len(self.classifiers_), self.n_classes_))
        self.precomputed_weights_ = np.zeros((len(inp), len(self.classifiers_)))
        for i, x in enumerate(inp):
            for j, cls in enumerate(self.classifiers_):
                prob = cls.predict_proba(x)[0]
                if not self.use_prob:
                    max_index = prob.argmax()
                    prob = np.zeros_like(prob)
                    prob[max_index] = 1
                self.precomputed_prob_[i, j] = prob
            self.precomputed_weights_[i] = self.weighting_strategy.weight_classifiers(x)
            if self.multiply_by_weight:
                for j in range(len(self.classifiers_)):
                    self.precomputed_prob_[i, j] *= self.precomputed_weights_[j]
            print "\rComputing", len(inp), "probabilities and weights:", (i+1),
        print ""