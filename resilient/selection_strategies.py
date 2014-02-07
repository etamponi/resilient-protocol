from abc import ABCMeta, abstractmethod

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SelectionStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def select(self, weights):
        pass


class SelectBestK(SelectionStrategy):

    def __init__(self, k=10):
        super(SelectBestK, self).__init__()
        self.k = k

    def select(self, weights):
        indices = weights.argsort()[::-1]
        return indices[:self.k]
