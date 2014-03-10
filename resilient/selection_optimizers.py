from abc import ABCMeta, abstractmethod
from itertools import product
import operator

import numpy
from scipy.ndimage import convolve
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from resilient.logger import Logger


__author__ = 'tamponi'


class SelectionOptimizer(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, ensemble, inp, y):
        pass


class GridOptimizer(SelectionOptimizer):

    def __init__(self, steps=500, kernel_size=5, scoring=accuracy_score, custom_ranges=None):
        self.steps = steps
        self.kernel_size = kernel_size
        self.scoring = scoring
        self.custom_ranges = custom_ranges

    def optimize(self, ensemble, inp, y):
        keys, params = self.build_params_matrix(ensemble.selection_strategy)
        scores = numpy.zeros(params.shape[:-1])
        for index in xrange(reduce(operator.mul, scores.shape, 1)):
            index = numpy.unravel_index(index, scores.shape)
            curr_param = params[index]
            param_dict = {key: curr_param[i] for i, key in enumerate(keys)}
            ensemble.selection_strategy.params = param_dict
            Logger.get().write("!Optimization - parameters: {}".format(
                ensemble.selection_strategy.params_to_string(join=" ")
            ))
            scores[index] = self.scoring(y, ensemble.predict(inp))
        kernel = numpy.ones(((self.kernel_size,) * len(scores.shape)))
        kernel /= kernel.sum()
        averaged_scores = convolve(scores, kernel)
        best_index = numpy.unravel_index(averaged_scores.argmax(), averaged_scores.shape)
        best_param = params[best_index]
        best_param = {key: best_param[i] for i, key in enumerate(keys)}
        ensemble.selection_strategy.params = best_param
        Logger.get().write("Selected parameters: [{}] with score {:.3f} (averaged {:.3f})".format(
            ensemble.selection_strategy.params_to_string(join=" "), scores[best_index], averaged_scores[best_index]
        ))

    def build_params_matrix(self, selection_strategy, matrix_form=True):
        custom_ranges = self.custom_ranges if self.custom_ranges is not None else {}
        keys = selection_strategy.get_params_names()
        ranges = [
            custom_ranges[key] if key in custom_ranges else numpy.linspace(0, 1, self.steps+1) for key in keys
        ]
        # indices = product(*[range(len(r)) for r in ranges])
        params = numpy.array(list(product(*ranges)))
        if matrix_form:
            params = numpy.reshape(params, tuple(len(r) for r in ranges) + (len(params[0]),))
        return keys, params
