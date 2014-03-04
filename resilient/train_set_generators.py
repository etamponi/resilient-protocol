from cmath import log
from abc import ABCMeta, abstractmethod
from math import floor

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator
from sklearn.cluster.k_means_ import MiniBatchKMeans

from resilient.pdfs import DistanceExponential


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class TrainSetGenerator(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_indices(self, inp, y, random_state):
        pass


class CentroidBasedPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, n_estimators=101, pdf=DistanceExponential(), percent=0.35, replace=False, repeat=False):
        self.n_estimators = n_estimators
        self.pdf = pdf
        self.percent = percent
        self.replace = replace
        self.repeat = repeat

    def get_indices(self, inp, y, random_state):
        print "\rTraining", self.n_estimators, "estimators..."
        for probs in self._get_probabilities(inp, random_state):
            yield self._make_indices(len(inp), probs, random_state)

    def _make_indices(self, l, probs, random_state):
        indices = random_state.choice(l, size=int(self.percent*l), p=probs, replace=self.replace)
        if not self.repeat:
            indices = numpy.unique(indices)
        return indices

    def _get_probabilities(self, inp, random_state):
        mean_probs = numpy.ones(inp.shape[0]) / inp.shape[0]
        for i in range(self.n_estimators):
            mean = inp[random_state.choice(len(inp), p=mean_probs)]
            probs = self.pdf.probabilities(inp, mean=mean)
            for j, x in enumerate(inp):
                mean_probs[j] *= log(1 + distance.euclidean(x, mean)).real
            mean_probs = mean_probs / mean_probs.sum()
            yield probs


class GridPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, n_estimators=None, spacing=0.5, pdf=DistanceExponential(),
                 percent=1.0, replace=True, repeat=True):
        self.n_estimators = n_estimators
        self.spacing = spacing
        self.pdf = pdf
        self.percent = percent
        self.replace = replace
        self.repeat = repeat

    def get_indices(self, inp, y, random_state):
        cells = self._get_cells(inp)
        for probs in self._get_probabilities(inp, cells, random_state):
            yield self._make_indices(len(inp), probs, random_state)

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
        print "\rTraining", len(cells), "estimators..."
        for cell in cells:
            mean = (numpy.array(cell) + 0.5) * self.spacing
            probs = self.pdf.probabilities(inp, mean=mean)
            yield probs

    def _make_indices(self, size, probs, random_state):
        indices = random_state.choice(size, size=int(self.percent*size), p=probs, replace=self.replace)
        if not self.repeat:
            indices = numpy.unique(indices)
        return indices


class ClusteringPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, clustering=MiniBatchKMeans(n_clusters=51),
                 pdf=DistanceExponential(), percent=1.0, replace=True, repeat=True):
        self.clustering = clustering
        self.pdf = pdf
        self.percent = percent
        self.replace = replace
        self.repeat = repeat

    def get_indices(self, inp, y, random_state):
        self.clustering.set_params(random_state=random_state)
        self.clustering.fit(inp)
        for probs in self._get_probabilities(inp):
            yield self._make_indices(len(inp), probs, random_state)

    def _get_probabilities(self, inp):
        centroids = self.clustering.cluster_centers_
        print "\rTraining", len(centroids), "estimators..."
        for centroid in centroids:
            yield self.pdf.probabilities(inp, mean=centroid)

    def _make_indices(self, size, probs, random_state):
        indices = random_state.choice(size, size=int(self.percent*size), p=probs, replace=self.replace)
        if not self.repeat:
            indices = numpy.unique(indices)
        return indices
