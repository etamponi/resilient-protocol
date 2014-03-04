from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.spatial.distance
from sklearn.base import BaseEstimator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class WeightingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare(self, inp, y):
        pass

    @abstractmethod
    def add_estimator(self, est, train_set, validation_set):
        pass

    @abstractmethod
    def weight_classifiers(self, x):
        pass


class CentroidBasedWeightingStrategy(WeightingStrategy):

    def __init__(self, dist_measure=scipy.spatial.distance.euclidean):
        self.dist_measure = dist_measure
        self.centroids_ = None

    def prepare(self, inp, y):
        self.centroids_ = []

    def add_estimator(self, est, train_set, validation_set):
        self.centroids_.append(train_set.data.mean(axis=0))

    def weight_classifiers(self, x):
        scores = np.array([1 / self.dist_measure(x, centroid) for centroid in self.centroids_])
        return scores
