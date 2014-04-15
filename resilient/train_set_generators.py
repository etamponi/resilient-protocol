from abc import ABCMeta, abstractmethod
import cmath
import numpy
from scipy.spatial import distance

from sklearn.base import BaseEstimator
from sklearn.cluster.k_means_ import MiniBatchKMeans, KMeans

from resilient.logger import Logger
from resilient.pdfs import DistanceExponential


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class TrainSetGenerator(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_sample_weights(self, n_estimators, inp, y, random_state):
        pass


class RandomCentroidPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, pdf=DistanceExponential()):
        self.pdf = pdf

    def get_sample_weights(self, n_estimators, inp, y, random_state):
        Logger.get().write("!Training", n_estimators, "estimators...")
        mean_probs = numpy.ones(inp.shape[0]) / inp.shape[0]
        for i in xrange(n_estimators):
            mean = inp[random_state.choice(len(inp), p=mean_probs)]
            probs = self.pdf.probabilities(inp, mean=mean)
            for j, x in enumerate(inp):
                mean_probs[j] *= cmath.log(1 + distance.euclidean(x, mean)).real
            mean_probs = mean_probs / mean_probs.sum()
            yield probs


class ClusterAlgorithmWrapper(BaseEstimator):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, inp, n_clusters, random_state):
        pass
    
    @abstractmethod
    def get_centroids(self):
        pass


class KMeansWrapper(ClusterAlgorithmWrapper):
    
    def __init__(self, max_iter=300, use_mini_batch=True):
        self.max_iter = max_iter
        self.use_mini_batch = use_mini_batch
        self.algorithm_ = None
    
    def fit(self, inp, n_clusters, random_state):
        if self.use_mini_batch:
            self.algorithm_ = MiniBatchKMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iter,
                random_state=random_state
            ).fit(inp)
        else:
            self.algorithm_ = KMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iter,
                random_state=random_state
            ).fit(inp)
    
    def get_centroids(self):
        return self.algorithm_.cluster_centers_


class ClusteringPDFTrainSetGenerator(TrainSetGenerator):

    def __init__(self, clustering=KMeansWrapper(), pdf=DistanceExponential()):
        self.clustering = clustering
        self.pdf = pdf

    def get_sample_weights(self, n_estimators, inp, y, random_state):
        self.clustering.fit(inp, n_estimators, random_state)
        centroids = self.clustering.get_centroids()
        Logger.get().write("!Training", len(centroids), "estimators...")
        for centroid in centroids:
            yield self.pdf.probabilities(inp, mean=centroid)
