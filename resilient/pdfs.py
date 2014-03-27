from abc import ABCMeta, abstractmethod
import cmath

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class PDF(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def probability(self, x):
        pass

    def probabilities(self, inp, **runtime_args):
        old_values = {}
        for key, value in runtime_args.iteritems():
            if not hasattr(self, key):
                continue
            old_values[key] = getattr(self, key)
            setattr(self, key, value)

        probs = numpy.array([self.probability(x) for x in inp])
        if probs.min() < 0:
            probs -= probs.min()
        probs = probs / probs.sum()

        for key, value in old_values.iteritems():
            setattr(self, key, value)
        return probs


class DistanceNormal(PDF):

    def __init__(self, mean=0, precision=20, base=cmath.e, sqdist_measure="sqeuclidean"):
        self.mean = mean
        self.precision = precision
        self.base = base
        self.sqdist_measure = sqdist_measure

    def probability(self, x):
        return (self.base**(-0.5 * self.precision * self._sqdist_measure(x, self.mean))).real

    def _sqdist_measure(self, u, v):
        return getattr(distance, self.sqdist_measure)(u, v)


class DistanceExponential(PDF):

    def __init__(self, mean=0, tau=0.15, base=cmath.e, dist_measure="euclidean"):
        self.mean = mean
        self.base = base
        self.tau = tau
        self.dist_measure = dist_measure

    def probability(self, x):
        return (self.base**(-self._dist_measure(x, self.mean) / self.tau)).real

    def _dist_measure(self, u, v):
        return getattr(distance, self.dist_measure)(u, v)


class DistanceGeneralExp(PDF):

    def __init__(self, mean=0, precision=0, base=cmath.e, power=2):
        self.mean = mean
        self.base = base
        self.precision = precision
        self.power = power

    def probability(self, x):
        return (self.base**(-0.5 * self.precision * distance.euclidean(x, self.mean)**self.power)).real


class DistanceInverse(PDF):

    def __init__(self, mean=0, power=1, dist_measure="euclidean"):
        self.mean = mean
        self.power = power
        self.dist_measure = dist_measure

    def probability(self, x):
        return 1.0 / (self._dist_measure(x, self.mean) + 1)**self.power

    def _dist_measure(self, u, v):
        return getattr(distance, self.dist_measure)(u, v)


class Uniform(PDF):

    def probability(self, x):
        return 1.0


class DistancePower(PDF):

    def __init__(self, mean=0, power=1):
        self.mean = mean
        self.power = power

    def probability(self, x):
        return -distance.euclidean(x, self.mean)**self.power
