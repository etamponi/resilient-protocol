from abc import ABCMeta, abstractmethod

import numpy
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


__author__ = 'tamponi'


class SelectionOptimizer(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, ensemble, inp, y):
        pass


class SimpleOptimizer(SelectionOptimizer):

    def __init__(self, kernel=numpy.ones(5)/5, scoring=accuracy_score):
        self.kernel = kernel
        self.scoring = scoring

    def optimize(self, ensemble, inp, y):
        original_param = ensemble.selection_strategy.param
        params = []
        scores = []
        for param in ensemble.selection_strategy.get_param_set(ensemble):
            print "\rOptimization - Checking parameter: {:.3f}".format(param),
            ensemble.selection_strategy.param = param
            params.append(param)
            scores.append(self.scoring(y, ensemble.predict(inp)))
        print ""
        averaged_scores = numpy.convolve(scores, self.kernel, "same")
        best_index = averaged_scores.argmax()
        best_param = params[best_index]
        print "Selected parameter: {:.3f} with score {:.3f} (averaged {:.3f})".format(best_param, scores[best_index],
                                                                                      averaged_scores[best_index])
        ensemble.selection_strategy.param = best_param
        return original_param
