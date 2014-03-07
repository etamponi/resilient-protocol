from abc import ABCMeta, abstractmethod

import numpy

from numpy.core.function_base import linspace
from sklearn.base import BaseEstimator
from resilient.logger import Logger


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, item):
        return self.params[item]

    def __setattr__(self, key, value):
        if key != "params" and key in self.params:
            self.params[key] = value
        else:
            super(SelectionStrategy, self).__setattr__(key, value)

    @abstractmethod
    def get_indices(self, weights):
        """
        Returns the indices of the corresponding classifiers, using their weights.
        """
        pass

    @abstractmethod
    def get_params_ranges(self):
        """
        Returns a dict of arrays, each containing the possible values for the corresponding param
        """
        pass

    def get_params_names(self):
        return sorted(self.params.keys())

    def params_to_string(self, params=None, join=None):
        ret = []
        if params is None:
            params = self.params
        for key in self.get_params_names():
            ret.append(Logger.format_number(params[key]))
        if join is not None:
            ret = join.join(ret)
        return ret


class SelectBestPercent(SelectionStrategy):

    def __init__(self, percent=0.10, steps=50):
        super(SelectBestPercent, self).__init__(percent=percent)
        self.steps = steps

    def get_indices(self, weights):
        indices = weights.argsort()
        k = round(self.percent * len(weights))
        k = 1 if k < 1 else k
        # Higher values at the end of the list
        return indices[-k:]

    def get_params_ranges(self):
        return {"percent": linspace(0, 1, self.steps+1)}


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, threshold=0.10, steps=500):
        super(SelectByWeightSum, self).__init__(threshold=threshold)
        self.steps = steps

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.threshold:
                return indices[:k+1]
        return indices

    def get_params_ranges(self):
        return {"threshold": linspace(0, 1, self.steps+1)[1:]}


class SelectByWeightThreshold(SelectionStrategy):

    def __init__(self, threshold=0.10, steps=500):
        super(SelectByWeightThreshold, self).__init__(threshold=threshold)
        self.steps = steps

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        for k in xrange(len(indices)):
            if weights[indices[k]] < self.threshold:
                return indices[:k]
        return indices

    def get_params_ranges(self):
        return {"threshold": linspace(0, 1, self.steps+1)[1:]}


class SelectSkippingNearHypersphere(SelectionStrategy):

    def __init__(self, similarity=0.01, inner_strategy=SelectBestPercent(), max_similarity=0.2, steps=20):
        inner_params = {"inner_" + key: value for key, value in inner_strategy.params.iteritems()}
        super(SelectSkippingNearHypersphere, self).__init__(similarity=similarity, **inner_params)
        self.inner_strategy = inner_strategy
        self.max_similarity = max_similarity
        self.steps = steps

    def get_indices(self, weights):
        inner_params = {key[6:]: value for key, value in self.params.iteritems() if key.startswith("inner_")}
        indices = weights.argsort()[::-1]
        filtered_weights = numpy.zeros(len(weights))
        filtered_weights[indices[0]] = weights[indices[0]]
        last_weight = filtered_weights[indices[0]]
        for i in indices[1:]:
            if self._diff_percent(last_weight, weights[i]) >= self.similarity:
                filtered_weights[i] = weights[i]
                last_weight = filtered_weights[i]
        self.inner_strategy.params = inner_params
        return self.inner_strategy.get_indices(filtered_weights)

    def get_params_ranges(self):
        ret = {"similarity": linspace(0, self.max_similarity, self.steps+1)}
        inner = {"inner_" + key: value for key, value in self.inner_strategy.get_params_ranges().iteritems()}
        ret.update(inner)
        return ret

    @staticmethod
    def _diff_percent(a, b):
        return 2.0 * abs(a - b) / (a + b)
