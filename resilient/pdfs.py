from abc import ABCMeta, abstractmethod
from cmath import exp

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class PDF(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def probability(self, x, **runtime_args):
        pass

    def probabilities(self, inp, **runtime_args):
        p = numpy.zeros(inp.shape[0])
        for i, x in enumerate(inp):
            p[i] = self.probability(x, **runtime_args)
        p = p / p.sum()
        return p


class DistanceNormal(PDF):

    def __init__(self, precision=20, sqdist_measure=distance.sqeuclidean):
        self.precision = precision
        self.sqdist_measure = sqdist_measure

    def probability(self, x, mean=None):
        return exp(0.5 * self.precision * self.sqdist_measure(x, mean)).real


class DistanceExponential(PDF):

    def __init__(self, base=exp(1), tau=0.15, dist_measure=distance.euclidean):
        self.base = base
        self.tau = tau
        self.dist_measure = dist_measure

    def probability(self, x, mean=None):
        return self.base**(-self.dist_measure(x, mean) / self.tau)


class Uniform(PDF):

    def probability(self, x, **runtime_args):
        return 1.0
