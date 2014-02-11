from abc import ABCMeta, abstractmethod
import numpy
import scipy.spatial.distance
from sklearn.base import BaseEstimator

from resilient.dataset import Dataset
from resilient.pdfs import MultivariateExponential


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SplittingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def iterate(self, n_estimators, inp, y, random_state):
        pass

    @staticmethod
    def _make_datasets(inp, y, train_indices, test_indices):
        train_data, test_data = inp[train_indices], inp[test_indices]
        train_targ, test_targ = y[train_indices], y[test_indices]
        return Dataset(train_data, train_targ), Dataset(test_data, test_targ)


class CentroidBasedPDFSplittingStrategy(SplittingStrategy):

    def __init__(self, pdf=MultivariateExponential(), train_percent=0.35, replace=False, repeat=False):
        self.pdf = pdf
        self.train_percent = train_percent
        self.replace = replace
        self.repeat = repeat

    def iterate(self, n_estimators, inp, y, random_state):
        for probs in self._get_probabilities(inp, n_estimators, random_state):
            train_indices, test_indices = self._make_indices(len(inp), probs, random_state)
            yield self._make_datasets(inp, y, train_indices, test_indices)

    def _make_indices(self, l, pdf, random_state):
        train_indices = random_state.choice(l, size=int(self.train_percent*l), p=pdf, replace=self.replace)
        if not self.repeat:
            train_indices = numpy.unique(train_indices)
        test_indices = numpy.ones(l, dtype=bool)
        test_indices[train_indices] = False
        return train_indices, test_indices

    def _get_probabilities(self, inp, n_means, random_state):
        mean_prob = numpy.ones(inp.shape[0]) / inp.shape[0]
        for i in range(n_means):
            mean = inp[random_state.choice(len(inp), p=mean_prob)]
            probs = self.pdf.probabilities(inp, mean=mean)
            for j, x in enumerate(inp):
                mean_prob[j] *= scipy.spatial.distance.sqeuclidean(x, mean)
            mean_prob = mean_prob / mean_prob.sum()
            yield probs


class CentroidBasedKNNSplittingStrategy(SplittingStrategy):

    def __init__(self, train_percent=0.25):
        self.train_percent = train_percent

    def iterate(self, n_estimators, inp, y, random_state):
        k = int(self.train_percent * len(inp))
        for mean, distances in self._get_means(inp, n_estimators, random_state):
            indices = distances.argsort()
            train_indices, test_indices = indices[:k], indices[k:]
            yield self._make_datasets(inp, y, train_indices, test_indices)

    @staticmethod
    def _get_means(inp, n_means, random_state):
        mean_prob = numpy.ones(inp.shape[0]) / inp.shape[0]
        distances = numpy.zeros(inp.shape[0])
        for i in range(n_means):
            mean = inp[random_state.choice(len(inp), p=mean_prob)]
            for j, x in enumerate(inp):
                distances[j] = scipy.spatial.distance.euclidean(x, mean)
                mean_prob[j] *= distances[j]
            mean_prob = mean_prob / mean_prob.sum()
            yield mean, distances
