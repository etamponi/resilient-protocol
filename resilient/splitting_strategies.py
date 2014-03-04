from cmath import log
from abc import ABCMeta, abstractmethod
from math import floor

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator

from resilient.dataset import Dataset
from resilient.pdfs import DistanceExponential


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SplittingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def iterate(self, inp, y, random_state):
        pass

    @staticmethod
    def _make_datasets(inp, y, train_indices, test_indices):
        train_data, test_data = inp[train_indices], inp[test_indices]
        train_targ, test_targ = y[train_indices], y[test_indices]
        return Dataset(train_data, train_targ), Dataset(test_data, test_targ)


class CentroidBasedPDFSplittingStrategy(SplittingStrategy):

    def __init__(self, n_estimators=101, pdf=DistanceExponential(), train_percent=0.35, replace=False, repeat=False):
        self.n_estimators = n_estimators
        self.pdf = pdf
        self.train_percent = train_percent
        self.replace = replace
        self.repeat = repeat

    def iterate(self, inp, y, random_state):
        print "\rTraining", self.n_estimators, "estimators..."
        for probs, centroid in self._get_probabilities(inp, random_state):
            train_indices = self._make_indices(len(inp), probs, random_state)
            yield self._make_datasets(inp, y, train_indices, train_indices), centroid

    def _make_indices(self, l, probs, random_state):
        train_indices = random_state.choice(l, size=int(self.train_percent*l), p=probs, replace=self.replace)
        if not self.repeat:
            train_indices = numpy.unique(train_indices)
        return train_indices

    def _get_probabilities(self, inp, random_state):
        mean_probs = numpy.ones(inp.shape[0]) / inp.shape[0]
        for i in range(self.n_estimators):
            mean = inp[random_state.choice(len(inp), p=mean_probs)]
            probs = self.pdf.probabilities(inp, mean=mean)
            for j, x in enumerate(inp):
                mean_probs[j] *= log(1 + distance.euclidean(x, mean)).real
            mean_probs = mean_probs / mean_probs.sum()
            yield probs, mean


class GridPDFSplittingStrategy(SplittingStrategy):

    def __init__(self, n_estimators=None, spacing=0.5, pdf=DistanceExponential(),
                 train_percent=1.0, replace=True, repeat=True):
        self.n_estimators = n_estimators
        self.spacing = spacing
        self.pdf = pdf
        self.train_percent = train_percent
        self.replace = replace
        self.repeat = repeat

    def iterate(self, inp, y, random_state):
        cells = self._get_cells(inp)
        for probs, centroid in self._get_probabilities(inp, cells, random_state):
            train_indices = self._make_indices(len(inp), probs, random_state)
            yield self._make_datasets(inp, y, train_indices, train_indices), centroid

    def _get_cells(self, inp):
        cells = set([])
        for x in inp:
            code = tuple([floor(t / self.spacing) for t in x])
            cells.add(code)
        print "Cells found:", len(cells)
        return list(cells)

    def _get_probabilities(self, inp, cells, random_state):
        cells = random_state.permutation(cells)
        if self.n_estimators is not None and self.n_estimators < len(cells):
            cells = cells[:self.n_estimators]
        for cell in cells:
            mean = (numpy.array(cell) + 0.5) * self.spacing
            probs = self.pdf.probabilities(inp, mean=mean)
            yield probs, mean

    def _make_indices(self, l, probs, random_state):
        train_indices = random_state.choice(l, size=int(self.train_percent*l), p=probs, replace=self.replace)
        if not self.repeat:
            train_indices = numpy.unique(train_indices)
        return train_indices
