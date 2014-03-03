from abc import ABCMeta, abstractmethod

from numpy.core.numeric import asarray
from numpy.core.umath import sign
import numpy as np
import scipy.spatial.distance
from sklearn.base import BaseEstimator
from sklearn.metrics.metrics import accuracy_score
from sklearn.utils.validation import array2d


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class WeightingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare(self, inp, y):
        pass

    @abstractmethod
    def add_estimator(self, est, train_set, test_set):
        pass

    @abstractmethod
    def weight_classifiers(self, x):
        pass


class LocalScoreWeightingStrategy(WeightingStrategy):

    def __init__(self, k=15, scoring=accuracy_score):
        self.k = k
        self.scoring = scoring
        # Training time attributes
        self.classifiers_ = None

    def prepare(self, inp, y):
        self.classifiers_ = []

    def add_estimator(self, est, train_set, test_set):
        self.classifiers_.append((est, test_set))
        return self

    def weight_classifiers(self, x):
        scores = np.zeros(len(self.classifiers_))
        for i, (cls, ds) in enumerate(self.classifiers_):
            inp, y = self._get_neighborhood(x, ds)
            y_pred = cls.predict(inp)
            try:
                scores[i] = self.scoring(y, y_pred)
            except ValueError:
                pass
        return scores

    def _get_neighborhood(self, x, dataset):
        sorting = []
        for inst, targ in dataset:
            sorting.append((scipy.spatial.distance.sqeuclidean(x, inst), inst, targ))
        sorting.sort(cmp=lambda a, b: int(sign(a[0] - b[0])))
        inp = [sample[1] for sample in sorting[:self.k]]
        y = [sample[2] for sample in sorting[:self.k]]
        return array2d(inp), asarray(y)


class CentroidBasedWeightingStrategy(WeightingStrategy):

    def __init__(self, dist_measure=scipy.spatial.distance.euclidean, use_real_centroid=True):
        self.dist_measure = dist_measure
        self.use_real_centroid = use_real_centroid
        self.centroids_ = None

    def prepare(self, inp, y):
        self.centroids_ = []

    def add_estimator(self, est, train_set, test_set):
        if self.use_real_centroid:
            self.centroids_.append(train_set.data.mean(axis=0))
        else:
            self.centroids_.append(est.centroid_)

    def weight_classifiers(self, x):
        scores = np.array([1 / self.dist_measure(x, centroid) for centroid in self.centroids_])
        return scores
