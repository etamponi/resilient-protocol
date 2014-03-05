from abc import ABCMeta, abstractmethod
from itertools import product
import numpy

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


class SelectSkipping(SelectionStrategy):

    def __init__(self, param=(0.01, 10), selection=SelectBestK(), max_percent=0.2, steps=20):
        super(SelectSkipping, self).__init__(param)
        self.selection = selection
        self.max_percent = max_percent
        self.steps = steps

    def get_indices(self, weights):
        percent, inner_param = self.param
        selection_class = self.selection.__class__
        indices = weights.argsort()[::-1]
        rest = numpy.zeros(len(weights))
        rest[indices[0]] = weights[indices[0]]
        last_weight = rest[indices[0]]
        for i in indices[1:]:
            if self._diff_percent(last_weight, weights[i]) >= percent:
                rest[i] = weights[i]
                last_weight = rest[i]
        return selection_class(param=inner_param).get_indices(rest)

    def get_param_set(self, ensemble):
        return list(product(linspace(0, self.max_percent, self.steps), self.selection.get_param_set(ensemble)))

    @staticmethod
    def _diff_percent(a, b):
        return 2.0 * abs(a - b) / (a + b)
