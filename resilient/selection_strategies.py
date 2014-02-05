from abc import ABCMeta, abstractmethod

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def select(self, sorted_classifiers, weights):
        pass


class SelectBestK(SelectionStrategy):

    def __init__(self, k=10):
        super(SelectBestK, self).__init__()
        self.k = k

    def select(self, sorted_classifiers, weights):
        return sorted_classifiers[:self.k], weights[:self.k]


class SelectWithThreshold(SelectionStrategy):

    def __init__(self, threshold=0.5):
        super(SelectWithThreshold, self).__init__()
        self.threshold = threshold

    def select(self, sorted_classifiers, weights):
        weights = [w for w in weights if w >= self.threshold]
        return sorted_classifiers[:len(weights)], weights
