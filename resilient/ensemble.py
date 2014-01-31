import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.fixes import unique
from sklearn.utils import check_random_state

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class ResilientEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state
        # Training time attributes
        self.random_state_ = None
        self.classes_ = None

    def fit(self, inp, y):
        self.random_state_ = check_random_state(self.random_state)
        self.classes_, y = unique(y, return_inverse=True)
        # Continue here

    def predict_proba(self, inp):
        return np.tile([1, 0], (len(inp), 1))

    def predict(self, inp):
        p = self.predict_proba(inp)
        return self.classes_[np.argmax(p, axis=1)]
