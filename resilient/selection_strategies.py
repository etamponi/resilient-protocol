from abc import ABCMeta, abstractmethod
import numpy

from numpy.core.function_base import linspace
from sklearn.metrics.metrics import accuracy_score

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_indices(self, weights):
        """
        Returns the indices of the corrisponding classifiers, using their weights.
        """
        pass

    def optimize(self, ensemble, inp, y):
        params = []
        scores = []
        for param in self._optimization_grid(ensemble):
            print "\rOptimization - Checking parameter: {:.3f}".format(param),
            self._set_param(param)
            params.append(param)
            scores.append(accuracy_score(y, ensemble.predict(inp)))
        print ""
        averaged_scores = numpy.convolve(scores, numpy.ones(5)/5, "same")
        best_index = averaged_scores.argmax()
        best_param = params[best_index]
        print "Selected parameter: {:.3f} with score {:.3f} (averaged {:.3f})".format(best_param, scores[best_index],
                                                                                      averaged_scores[best_index])
        self._set_param(best_param)

    @abstractmethod
    def _optimization_grid(self, ensemble):
        return []

    @abstractmethod
    def _set_param(self, param):
        pass


class SelectBestK(SelectionStrategy):

    def __init__(self, k=10):
        self.k = k

    def get_indices(self, weights):
        indices = weights.argsort()
        # Higher values at the end of the list
        return indices[-self.k:]

    def _optimization_grid(self, ensemble):
        return range(1, len(ensemble.classifiers_)+1, 2)

    def _set_param(self, param):
        self.k = param


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.threshold:
                return indices[:k+1]
        return indices

    def _optimization_grid(self, ensemble):
        return linspace(0, 1, 501)[1:]

    def _set_param(self, param):
        self.threshold = param


class SelectByThreshold(SelectionStrategy):

    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        for k in xrange(len(indices)):
            if weights[indices[k]] < self.threshold:
                return indices[:k]
        return indices

    def _optimization_grid(self, ensemble):
        return linspace(0, 1, 501)[1:]

    def _set_param(self, param):
        self.threshold = param
