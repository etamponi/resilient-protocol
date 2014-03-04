from abc import ABCMeta, abstractmethod

import numpy

from numpy.core.function_base import linspace
from sklearn.base import BaseEstimator
from sklearn.metrics.metrics import accuracy_score


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, param, kernel=numpy.ones(5)/5):
        self.param = param
        self.kernel = kernel

    @abstractmethod
    def get_indices(self, weights):
        """
        Returns the indices of the corrisponding classifiers, using their weights.
        """
        pass

    def optimize(self, ensemble, inp, y):
        params = []
        scores = []
        for param in self.get_optimization_grid(ensemble):
            print "\rOptimization - Checking parameter: {:.3f}".format(param),
            self.param = param
            params.append(param)
            scores.append(accuracy_score(y, ensemble.predict(inp)))
        print ""
        averaged_scores = numpy.convolve(scores, self.kernel, "same")
        best_index = averaged_scores.argmax()
        best_param = params[best_index]
        print "Selected parameter: {:.3f} with score {:.3f} (averaged {:.3f})".format(best_param, scores[best_index],
                                                                                      averaged_scores[best_index])
        self.param = best_param

    @abstractmethod
    def get_optimization_grid(self, ensemble):
        return []


class SelectBestK(SelectionStrategy):

    def __init__(self, param=10, kernel=numpy.ones(5)/5):
        super(SelectBestK, self).__init__(param, kernel)

    def get_indices(self, weights):
        indices = weights.argsort()
        # Higher values at the end of the list
        return indices[-self.param:]

    def get_optimization_grid(self, ensemble):
        return range(1, len(ensemble.classifiers_)+1, 2)


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, param=0.10, kernel=numpy.ones(5)/5):
        super(SelectByWeightSum, self).__init__(param, kernel)

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.param:
                return indices[:k+1]
        return indices

    def get_optimization_grid(self, ensemble):
        return linspace(0, 1, 501)[1:]


class SelectByThreshold(SelectionStrategy):

    def __init__(self, param=0.10, kernel=numpy.ones(5)/5):
        super(SelectByThreshold, self).__init__(param, kernel)

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        for k in xrange(len(indices)):
            if weights[indices[k]] < self.param:
                return indices[:k]
        return indices

    def get_optimization_grid(self, ensemble):
        return linspace(0, 1, 501)[1:]
