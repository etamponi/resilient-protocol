from abc import ABCMeta, abstractmethod
from cmath import exp

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
            old_values[key] = getattr(self, key)
            setattr(self, key, value)

        p = numpy.zeros(inp.shape[0])
        for i, x in enumerate(inp):
            p[i] = self.probability(x, **runtime_args)
        p = p / p.sum()

        for key, value in old_values:
            setattr(self, key, value)
        return p


class DistanceNormal(PDF):

    def __init__(self, mean=0, precision=20, base=exp(1), sqdist_measure=distance.sqeuclidean):
        self.mean = mean
        self.precision = precision
        self.base = base
        self.sqdist_measure = sqdist_measure

    def probability(self, x):
        return (self.base**(0.5 * self.precision * self.sqdist_measure(x, self.mean))).real


class DistanceExponential(PDF):

    def __init__(self, mean=0, tau=0.15, base=exp(1), dist_measure=distance.euclidean):
        self.mean = mean
        self.base = base
        self.tau = tau
        self.dist_measure = dist_measure

    def probability(self, x):
        return (self.base**(-self.dist_measure(x, self.mean) / self.tau)).real


class Uniform(PDF):

    def probability(self, x):
        return 1.0
