import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.fixes import unique
from sklearn import preprocessing

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class ResilientEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, training_strategy=None, selection_strategy=None, use_weights=False):
        self.training_strategy = training_strategy
        self.selection_strategy = selection_strategy
        self.use_weights = use_weights
        # Training time attributes
        self.classes_ = None
        self.n_classes_ = None
        self.classifiers_ = None

    def fit(self, inp, y):
        #self.random_state_ = check_random_state(self.random_state)
        self.classes_, y = unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.classifiers_ = self.training_strategy.train(inp, y)
        self.selection_strategy.fit(inp, y, classifiers=self.classifiers_)

    def predict_proba(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N, n_classes_), each row sums to one
        proba = np.zeros((len(inp), self.n_classes_))
        for i, x in enumerate(inp):
            active_proba, weights = self.selection_strategy.select(x)
            proba[i] = self._combine(active_proba, weights)
        preprocessing.normalize(proba, norm='l1', copy=False)
        return proba

    def _combine(self, weak_proba, weights):
        proba = np.zeros_like(weak_proba[0])
        for p, w in zip(weak_proba, weights):
            if self.use_weights:
                p *= w
            proba += p
        return proba

    def predict(self, inp):
        # inp is array-like, (N, D), one instance per row
        # output is array-like, (N), one label per instance
        p = self.predict_proba(inp)
        return self.classes_[np.argmax(p, axis=1)]
