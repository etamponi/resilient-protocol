from abc import ABCMeta, abstractmethod
import numpy

from sklearn.base import BaseEstimator
from sklearn.utils import array2d

from resilient import pdfs


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


class SelectRandomPercent(SelectionStrategy):

    def __init__(self, threshold=0.10, pdf=pdfs.DistanceExponential()):
        super(SelectRandomPercent, self).__init__(threshold)
        self.pdf = pdf

    def get_indices(self, weights, random_state):
        distances = array2d([[1 / w] for w in weights])
        p = self.pdf.probabilities(distances, mean=[0])
        k = int(round(self.threshold * len(distances)))
        k = 1 if k < 1 else k
        return random_state.choice(len(distances), size=k, p=p, replace=False)

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, n_estimators+1)[1:]


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, threshold=0.10):
        super(SelectByWeightSum, self).__init__(threshold)

    def get_indices(self, weights, random_state):
        weights = weights / sum(weights)
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
        weights = weights / sum(weights)
        indices = []
        for k, w in enumerate(weights):
            if weights[k] >= self.threshold:
                return indices.append(k)
        return indices

    def get_threshold_range(self, n_estimators):
        return numpy.linspace(0, 1, 10*n_estimators+1)[1:]
