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

    def __init__(self, mean=0, precision=20):
        self.mean = mean
        self.precision = precision

    def probability(self, x):
        return cmath.exp(
            -0.5 * self.precision * distance.sqeuclidean(x, self.mean)
        ).real


class DistanceExponential(PDF):

    def __init__(self, mean=0, tau=0.15):
        self.mean = mean
        self.tau = tau

    def probability(self, x):
        return cmath.exp(-distance.euclidean(x, self.mean) / self.tau).real


class DistanceGeneralizedExponential(PDF):

    def __init__(self, mean=0, precision=0, power=2):
        self.mean = mean
        self.precision = precision
        self.power = power

    def probability(self, x):
        return cmath.exp(
            -0.5 * self.precision * distance.euclidean(x, self.mean)**self.power
        ).real


class DistanceInverse(PDF):

    def __init__(self, mean=0, power=1):
        self.mean = mean
        self.power = power

    def probability(self, x):
        return (distance.euclidean(x, self.mean) + 1)**(-self.power)


class Uniform(PDF):

    def probability(self, x):
        return 1.0
