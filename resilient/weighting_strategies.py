from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.spatial import distance
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
    def weight_estimators(self, x):
        pass


class CentroidBasedWeightingStrategy(WeightingStrategy):

    def __init__(self, dist_measure=distance.euclidean):
        self.dist_measure = dist_measure
        self.centroids_ = None

    def prepare(self, inp, y):
        self.centroids_ = []

    def add_estimator(self, est, train_set, validation_set):
        self.centroids_.append(train_set.data.mean(axis=0))

    def weight_estimators(self, x):
        scores = np.array([1 / self.dist_measure(x, centroid) for centroid in self.centroids_])
        return scores


class CentroidShadowWeightingStrategy(WeightingStrategy):

    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.centroids_ = None

    def prepare(self, inp, y):
        self.centroids_ = []

    def add_estimator(self, est, train_set, validation_set):
        self.centroids_.append(train_set.data.mean(axis=0))

    # noinspection PyNoneFunctionAssignment
    def weight_estimators(self, x):
        distance_vectors = [x - centroid for centroid in self.centroids_]
        weights = np.array([1 / np.linalg.norm(v) for v in distance_vectors])
        final_weights = np.zeros_like(weights)
        while True:
            max_weight_index = weights.argmax()
            max_weight = weights[max_weight_index]
            if max_weight == 0.0:
                break
            final_weights[max_weight_index] = max_weight
            max_weight_vector = distance_vectors[max_weight_index]
            cosines = [distance.cosine(max_weight_vector, v) for v in distance_vectors]
            for i, cosine in enumerate(cosines):
                if cosine <= self.threshold:
                    weights[i] = 0.0
        return final_weights
