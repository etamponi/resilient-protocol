from abc import ABCMeta, abstractmethod

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_indices(self, weights):
        """
        Returns the indices of the corrisponding classifiers, using their weights.
        """
        pass


class SelectBestK(SelectionStrategy):

    def __init__(self, k=10):
        self.k = k

    def get_indices(self, weights):
        indices = weights.argsort()
        # Higher values at the end of the list
        return indices[-self.k:]


class SelectByWeightSum(SelectionStrategy):

    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def get_indices(self, weights):
        weights = weights / sum(weights)
        indices = weights.argsort()[::-1]
        partial_sum = 0
        for k in xrange(len(indices)):
            partial_sum += weights[indices[k]]
            if partial_sum >= self.threshold:
                return indices[:k+1]
        return indices
