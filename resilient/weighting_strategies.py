from abc import ABCMeta, abstractmethod

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator

from resilient import pdfs


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class WeightingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare(self, inp, y):
        pass

    @abstractmethod
    def add_estimator(self, est, inp, y, sample_weights):
        pass

    @abstractmethod
    def weight_estimators(self, x):
        pass


class SameWeight(WeightingStrategy):

    def __init__(self):
        self.weights_ = None

    def prepare(self, inp, y):
        self.weights_ = numpy.ones(0)

    def add_estimator(self, est, inp, y, sample_weights):
        self.weights_ = numpy.ones(1 + len(self.weights_))

    def weight_estimators(self, x):
        return self.weights_


class CentroidBasedWeightingStrategy(WeightingStrategy):

    def __init__(self, dist_measure="euclidean"):
        self.dist_measure = dist_measure
        self.centroids_ = None
        self.pdf_ = None

    def prepare(self, inp, y):
        self.centroids_ = []
        self.pdf_ = pdfs.DistanceInverse(power=5, dist_measure=self.dist_measure)

    def add_estimator(self, est, inp, y, sample_weights):
        self.centroids_.append(numpy.average(inp, axis=0, weights=sample_weights))

    def weight_estimators(self, x):
        scores = self.pdf_.probabilities(self.centroids_, mean=x)
        return scores

    def _dist_measure(self, u, v):
        return getattr(distance, self.dist_measure)(u, v)
