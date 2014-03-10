from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator
from sklearn.neighbors.ball_tree import array2d


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
            cosines = [
                distance.cosine(max_weight_vector, v) if weights[i] > 0 else 0 for i, v in enumerate(distance_vectors)
            ]
            for i, cosine in enumerate(cosines):
                if cosine <= self.threshold:
                    weights[i] = 0.0
        return final_weights


class CentroidRemovingNeighborsWeightingStrategy(WeightingStrategy):

    def __init__(self, k=3):
        self.k = k
        self.centroids_ = None
        self.neighbors_ = None

    def prepare(self, inp, y):
        self.centroids_ = []
        self.neighbors_ = None

    def add_estimator(self, est, train_set, validation_set):
        self.centroids_.append(train_set.data.mean(axis=0))

    def weight_estimators(self, x):
        if self.neighbors_ is None:
            self._prepare_neighbors()
        weights = np.array([1 / distance.euclidean(x, centroid) for centroid in self.centroids_])
        indices = list(weights.argsort()[::-1])
        already_in = set([])
        for i in indices:
            already_in.add(i)
            if weights[i] == 0.0:
                continue
            for j in (self.neighbors_[i] - already_in):
                weights[j] = 0.0
        return weights

    def _prepare_neighbors(self):
        self.centroids_ = array2d(self.centroids_)
        distances = distance.pdist(self.centroids_)
        if isinstance(self.k, float):
            k = int(distances.mean() * self.k)
        else:
            k = self.k
        distance_matrix = distance.squareform(distances)
        for distances in distance_matrix:
            indices = distances.argsort()
            self.neighbors_.append(set(indices[1:k+1]))
