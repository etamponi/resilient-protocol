from abc import ABCMeta, abstractmethod

import numpy
from sklearn.base import BaseEstimator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, threshold):
        self.threshold = threshold

    @abstractmethod
    def get_indices(self, weights, random_state):
        """
        Returns the indices of the corresponding classifiers, using their weights.
        """
        pass

    @abstractmethod
    def get_threshold_range(self, n_estimators):
        pass


class NoSelect(SelectionStrategy):

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, n_estimators+1)[1:]

    def get_indices(self, weights, random_state):
        pass


class SelectBestPercent(SelectionStrategy):

    def __init__(self, threshold=0.10):
        super(SelectBestPercent, self).__init__(threshold)

    def get_indices(self, weights, random_state):
        indices = weights.argsort()
        k = int(round(self.threshold * len(weights)))
        k = 1 if k < 1 else k
        # Higher values at the end of the list
        return indices[-k:]

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, n_estimators+1)[1:]


class SelectBestPercentSkipping(SelectionStrategy):

    def __init__(self, threshold=0.10, step=2):
        super(SelectBestPercentSkipping, self).__init__(threshold)
        self.step = step

    def get_indices(self, weights, random_state):
        indices = weights.argsort()[::-1]
        k = int(round(self.threshold * len(weights)) * self.step)
        k = 1 if k < 1 else k
        return indices[:k:self.step]

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1.0 / self.step, n_estimators + 1)[1:]


class SelectRandomPercent(SelectionStrategy):

    def __init__(self, threshold=0.10):
        super(SelectRandomPercent, self).__init__(threshold)

    def get_indices(self, weights, random_state):
        p = weights / weights.sum()
        k = int(round(self.threshold * len(weights)))
        k = 1 if k < 1 else k
        return random_state.choice(len(weights), size=k, p=p, replace=False)

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, n_estimators+1)[1:]


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, threshold=0.10):
        super(SelectByWeightSum, self).__init__(threshold)

    def get_indices(self, weights, random_state):
        weights = weights / weights.sum()
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.threshold:
                return indices[:k+1]
        return indices

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, 10*n_estimators+1)[1:]


class SelectByWeightThreshold(SelectionStrategy):

    def __init__(self, threshold=0.10):
        super(SelectByWeightThreshold, self).__init__(threshold)

    def get_indices(self, weights, random_state):
        weights = weights / weights.sum()
        indices = []
        for k, w in enumerate(weights):
            if weights[k] >= self.threshold:
                return indices.append(k)
        return indices

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, 10*n_estimators+1)[1:]
