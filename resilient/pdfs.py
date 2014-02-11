from abc import ABCMeta, abstractmethod
import numpy
import scipy.spatial.distance
from sklearn.base import BaseEstimator

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class PDF(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def probabilities(self, inp, **runtime_args):
        pass


class MultivariateNormal(PDF):

    def __init__(self, variance=0.05, sqdist_measure=scipy.spatial.distance.sqeuclidean):
        self.variance = variance
        self.sqdist_measure = sqdist_measure

    def probabilities(self, inp, mean=None, **runtime_args):
        p = numpy.zeros(inp.shape[0])
        for i, x in enumerate(inp):
            p[i] = 2**(-self.sqdist_measure(x, mean) / (2 * self.variance))
        p = p / p.sum()
        return p


class MultivariateExponential(PDF):

    def __init__(self, tau=0.15, dist_measure=scipy.spatial.distance.euclidean):
        self.tau = tau
        self.dist_measure = dist_measure

    def probabilities(self, inp, mean=None, **runtime_args):
        p = numpy.zeros(inp.shape[0])
        for i, x in enumerate(inp):
            p[i] = 2**(-self.dist_measure(x, mean) / self.tau)
        p = p / p.sum()
        return p
