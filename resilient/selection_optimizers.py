from abc import ABCMeta, abstractmethod
from itertools import product

import numpy
from scipy.ndimage import convolve
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


__author__ = 'tamponi'


class SelectionOptimizer(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, ensemble, inp, y):
        pass


class SimpleOptimizer(SelectionOptimizer):

    def __init__(self, kernel_size=5, scoring=accuracy_score):
        self.kernel_size = kernel_size
        self.scoring = scoring

    def optimize(self, ensemble, inp, y):
        indices, keys, params = self._build_params_matrix(ensemble.selection_strategy.get_params_ranges())
        scores = numpy.zeros(*params.shape[:-1])
        for index in indices:
            curr_param = params[index]
            param_dict = {key: curr_param[i] for i, key in enumerate(keys)}
            ensemble.selection_strategy.params = param_dict
            print "\rOptimization - parameters: {}".format(ensemble.selection_strategy.params_to_string(join=" ")),
            scores[index] = self.scoring(y, ensemble.predict(inp))
        print ""
        kernel = numpy.ones(*((self.kernel_size,) * len(scores.shape)))
        kernel /= sum(kernel)
        averaged_scores = convolve(scores, kernel)
        best_index = numpy.unravel_index(averaged_scores.argmax(), averaged_scores.shape)
        best_param = params[best_index]
        print "Selected parameters: {} with score {:.3f} (averaged {:.3f})".format(best_param, scores[best_index],
                                                                                   averaged_scores[best_index])
        best_param = {key: best_param[i] for i, key in enumerate(keys)}
        ensemble.selection_strategy.params = best_param

    @staticmethod
    def _build_params_matrix(ranges):
        keys = sorted(ranges.keys())
        ranges = [ranges[key] for key in keys]
        indices = product(*[range(len(r)) for r in ranges])
        params = numpy.array(list(product(*ranges)))
        params = numpy.reshape(params, tuple(len(r) for r in ranges) + (len(params[0]),))
        return indices, keys, params
