from abc import ABCMeta, abstractmethod

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def select(self, weights):
        """
        Returns the indices of the corrisponding classifiers, using their weights.
        """
        pass


class SelectBestK(SelectionStrategy):

    def __init__(self, k=10):
        super(SelectBestK, self).__init__()
        self.k = k

    def select(self, weights):
        indices = weights.argsort()
        return indices[-self.k:]
