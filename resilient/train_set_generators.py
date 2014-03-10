from cmath import log
from abc import ABCMeta, abstractmethod
from math import floor

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator
from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from sklearn.cluster.k_means_ import MiniBatchKMeans, KMeans
from sklearn.cluster.mean_shift_ import MeanShift
from sklearn.utils.validation import array2d

from resilient.logger import Logger

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
        Logger.get().write("!Training", self.n_estimators, "estimators...")
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
        Logger.get().write("Cells found:", len(cells))
        return list(cells)

    def _get_probabilities(self, inp, cells, random_state):
        cells = random_state.permutation(cells)
        if self.n_estimators is not None and self.n_estimators < len(cells):
            cells = cells[:self.n_estimators]
        Logger.get().write("!Training", len(cells), "estimators...")
        for cell in cells:
            mean = (numpy.array(cell) + 0.5) * self.spacing
            probs = self.pdf.probabilities(inp, mean=mean)
            yield probs

    def _make_indices(self, size, probs, random_state):
        indices = random_state.choice(size, size=int(self.percent*size), p=probs, replace=self.replace)
        if not self.repeat:
            indices = numpy.unique(indices)
        return indices
    
    
class ClusterAlgorithmWrapper(BaseEstimator):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, inp, random_state):
        pass
    
    @abstractmethod
    def get_centroids(self):
        pass
    
    
class KMeansWrapper(ClusterAlgorithmWrapper):
    
    def __init__(self, n_estimators=101, use_mini_batch=True):
        self.n_estimators = n_estimators
        self.use_mini_batch = use_mini_batch
        self.algorithm_ = None
    
    def fit(self, inp, random_state):
        if self.use_mini_batch:
            self.algorithm_ = MiniBatchKMeans(n_clusters=self.n_estimators, random_state=random_state).fit(inp)
        else:
            self.algorithm_ = KMeans(n_clusters=self.n_estimators, random_state=random_state).fit(inp)
    
    def get_centroids(self):
        return self.algorithm_.cluster_centers_
    

class AffinityPropagationWrapper(ClusterAlgorithmWrapper):
    
    def __init__(self):
        self.algorithm_ = None
        self.centroids_ = None
        
    def fit(self, inp, random_state):
        self.algorithm_ = AffinityPropagation().fit(inp)
        self.centroids_ = array2d([inp[i] for i in self.algorithm_.cluster_centers_indices_])

    def get_centroids(self):
        return self.centroids_
    
    
class MeanShiftWrapper(ClusterAlgorithmWrapper):
    
    def __init__(self):
        self.algorithm_ = None
    
    def get_centroids(self):
        return self.algorithm_.cluster_centers_

    def fit(self, inp, random_state):
        self.algorithm_ = MeanShift().fit(inp)


class ClusteringPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, clustering=KMeansWrapper(n_estimators=51),
                 pdf=DistanceExponential(), percent=1.0, replace=True, repeat=True):
        self.clustering = clustering
        self.pdf = pdf
        self.percent = percent
        self.replace = replace
        self.repeat = repeat

    def get_indices(self, inp, y, random_state):
        self.clustering.fit(inp, random_state)
        for probs in self._get_probabilities(inp):
            yield self._make_indices(len(inp), probs, random_state)

    def _get_probabilities(self, inp):
        centroids = self.clustering.get_centroids()
        Logger.get().write("!Training", len(centroids), "estimators...")
        for centroid in centroids:
            yield self.pdf.probabilities(inp, mean=centroid)

    def _make_indices(self, size, probs, random_state):
        indices = random_state.choice(size, size=int(self.percent*size), p=probs, replace=self.replace)
        if not self.repeat:
            indices = numpy.unique(indices)
        return indices
