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
        return (self.base**(0.5 * self.precision * self._sqdist_measure(x, self.mean))).real

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


class DistanceInverse(PDF):

    def __init__(self, mean=0, power=1, offset=0.01, dist_measure="euclidean"):
        self.mean = mean
        self.power = power
        self.offset = offset
        self.dist_measure = dist_measure

    def probability(self, x):
        return 1.0 / (self._dist_measure(x, self.mean) + self.offset)

    def _dist_measure(self, u, v):
        return getattr(distance, self.dist_measure)(u, v)


class Uniform(PDF):

    def probability(self, x):
        return 1.0
