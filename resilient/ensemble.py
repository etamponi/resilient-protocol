import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique
from sklearn import preprocessing
from sklearn.utils.random import check_random_state
from resilient.selection_strategies import SelectBestK
from resilient.sorting_strategies import LocalScoreSortingStrategy
from resilient.splitting_strategies import RandomChoiceSplittingStrategy
from resilient.wrapper import EstimatorWrapper

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

MAX_INT = np.iinfo(np.int32).max


class TrainingStrategy(object):

    def __init__(self,
                 n_estimators=100,
                 base_estimator=DecisionTreeClassifier(max_features='log2'),
                 splitting_strategy=RandomChoiceSplittingStrategy()):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.splitting_strategy = splitting_strategy

    def train_estimators(self, inp, y, random_state):
        classifiers = []
        for train_set, test_set in self.splitting_strategy.iterate(self.n_estimators, inp, y, random_state):
            est = self._make_estimator(random_state, train_set, test_set)
            classifiers.append(est)
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
                 sorting_strategy=LocalScoreSortingStrategy(),
                 selection_strategy=SelectBestK(),
                 use_weights=False,
                 random_state=None):
        self.training_strategy = training_strategy
        self.sorting_strategy = sorting_strategy
        self.selection_strategy = selection_strategy
        self.use_weights = use_weights
        self.random_state = random_state
        # Training time attributes
        self.classes_ = None
        self.n_classes_ = None
        self.classifiers_ = None
        self.random_state_ = None

    def fit(self, inp, y):
        self.classes_, y = unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.random_state_ = check_random_state(self.random_state)
        self.classifiers_ = self.training_strategy.train_estimators(inp, y, self.random_state_)
        self.sorting_strategy.prepare(self.classifiers_, self.random_state_)
        return self

    def predict_proba(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N, n_classes_), each row sums to one
        proba = np.zeros((len(inp), self.n_classes_))
        s = 0.0
        for i, x in enumerate(inp):
            sorted_classifiers, weights = self.sorting_strategy.sort_classifiers(x)
            active_classifiers, weights = self.selection_strategy.select(sorted_classifiers, weights)
            proba[i] = self._combine(x, active_classifiers, weights)
            s += len(active_classifiers)
        #print s / len(inp)
        preprocessing.normalize(proba, norm='l1', copy=False)
        return proba

    def predict(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N), one label per instance
        p = self.predict_proba(inp)
        return self.classes_[np.argmax(p, axis=1)]

    def _combine(self, x, active_classifiers, weights):
        proba = np.zeros(self.n_classes_)
        for c, w in zip(active_classifiers, weights):
            p = c.predict_proba(x)[0]
            if self.use_weights:
                p *= w
            proba += p
        return proba
