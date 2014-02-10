import scipy.spatial.distance

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


def squared_distance(a, b):
    return scipy.spatial.distance.sqeuclidean(a, b)
