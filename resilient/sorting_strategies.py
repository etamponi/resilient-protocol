from abc import ABCMeta, abstractmethod
from numpy.core.numeric import asarray
from numpy.core.umath import sign

import numpy as np
from sklearn.metrics.metrics import accuracy_score
from sklearn.utils.validation import array2d

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class WeightingStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def prepare(self, classifiers, random_state):
        pass

    @abstractmethod
    def weight_classifiers(self, x):
        pass


class LocalScoreWeightingStrategy(WeightingStrategy):

    def __init__(self, k=15, scoring=accuracy_score):
        super(LocalScoreWeightingStrategy, self).__init__()
        self.k = k
        self.scoring = scoring
        # Training time attributes
        self.classifiers_ = None

    def prepare(self, classifiers, random_state):
        self.classifiers_ = classifiers
        return self

    def weight_classifiers(self, x):
        scores = np.zeros(len(self.classifiers_))
        for i, cls in enumerate(self.classifiers_):
            inp, y = self._get_neighborhood(x, cls.test_set)
            y_pred = cls.predict(inp)
            try:
                scores[i] = self.scoring(y, y_pred)
            except ValueError:
                pass
        return scores

    def _get_neighborhood(self, x, dataset):
        sorting = []
        for inst, targ in dataset:
            sorting.append((np.linalg.norm(inst - x), inst, targ))
        sorting.sort(cmp=lambda a, b: int(sign(a[0] - b[0])))
        inp = [sample[1] for sample in sorting[:self.k]]
        y = [sample[2] for sample in sorting[:self.k]]
        return array2d(inp), asarray(y)


class CentroidBasedWeightingStrategy(WeightingStrategy):

    def __init__(self):
        super(CentroidBasedWeightingStrategy, self).__init__()
        self.centroids_ = None
        self.classifiers_ = None

    def prepare(self, classifiers, random_state):
        self.centroids_ = []
        self.classifiers_ = classifiers
        for cls in classifiers:
            self.centroids_.append(cls.train_set.data.mean(axis=0))

    def weight_classifiers(self, x):
        scores = np.array([1 / np.linalg.norm(x - centroid) for centroid in self.centroids_])
        return scores
