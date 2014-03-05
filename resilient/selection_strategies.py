from abc import ABCMeta, abstractmethod

from numpy.core.function_base import linspace
from sklearn.base import BaseEstimator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, param):
        self.param = param

    @abstractmethod
    def get_indices(self, weights):
        """
        Returns the indices of the corrisponding classifiers, using their weights.
        """
        pass

    @abstractmethod
    def get_param_set(self, ensemble):
        return []


class SelectBestK(SelectionStrategy):

    def __init__(self, param=10):
        super(SelectBestK, self).__init__(param)

    def get_indices(self, weights):
        indices = weights.argsort()
        # Higher values at the end of the list
        return indices[-self.param:]

    def get_param_set(self, ensemble):
        return range(1, len(ensemble.classifiers_)+1)


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, param=0.10):
        super(SelectByWeightSum, self).__init__(param)

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.param:
                return indices[:k+1]
        return indices

    def get_param_set(self, ensemble):
        return linspace(0, 1, 501)[1:]


class SelectByThreshold(SelectionStrategy):

    def __init__(self, param=0.10):
        super(SelectByThreshold, self).__init__(param)

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        for k in xrange(len(indices)):
            if weights[indices[k]] < self.param:
                return indices[:k]
        return indices

    def get_param_set(self, ensemble):
        return linspace(0, 1, 501)[1:]
