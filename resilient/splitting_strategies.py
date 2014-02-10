from cmath import exp

from abc import ABCMeta, abstractmethod
import numpy
from sklearn.cross_validation import Bootstrap, StratifiedShuffleSplit, ShuffleSplit

from resilient.dataset import Dataset
from resilient.utils import squared_distance


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SplittingStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def iterate(self, n_iter, inp, y, random_state):
        pass


class RandomChoiceSplittingStrategy(SplittingStrategy):

    def __init__(self, train_percent=0.7, replace=False, stratified=False):
        super(RandomChoiceSplittingStrategy, self).__init__()
        self.train_percent = train_percent
        self.replace = replace
        self.stratified = stratified

    def iterate(self, n_iter, inp, y, random_state):
        if self.replace and self.stratified:
            print "Warning! Cannot do a stratified bootstrap, doing a simple bootstrap instead"
        if self.replace:
            cv = Bootstrap(len(inp), n_iter=n_iter, train_size=self.train_percent, random_state=random_state)
        elif self.stratified:
            cv = StratifiedShuffleSplit(y, n_iter=n_iter, train_size=self.train_percent, random_state=random_state)
        else:
            cv = ShuffleSplit(len(inp), n_iter=n_iter, train_size=self.train_percent, random_state=random_state)
        for train_indices, test_indices in cv:
            train_indices, test_indices = numpy.unique(train_indices), numpy.unique(test_indices)
            train_data, test_data = inp[train_indices], inp[test_indices]
            train_targ, test_targ = y[train_indices], y[test_indices]
            yield Dataset(train_data, train_targ), Dataset(test_data, test_targ)


class CentroidBasedSplittingStrategy(SplittingStrategy):

    def __init__(self, variance=1, train_percent=0.7, replace=True, repeat=True):
        super(CentroidBasedSplittingStrategy, self).__init__()
        self.variance = variance
        self.train_percent = train_percent
        self.replace = replace
        self.repeat = repeat

    def iterate(self, n_iter, inp, y, random_state):
        for mean, p in self._get_means_and_p(inp, n_iter, random_state):
            train_indices, test_indices = self._make_indices(len(inp), p, random_state)
            train_data, test_data = inp[train_indices], inp[test_indices]
            train_targ, test_targ = y[train_indices], y[test_indices]
            yield Dataset(train_data, train_targ), Dataset(test_data, test_targ)

    def _make_indices(self, l, p, random_state):
        train_indices = random_state.choice(l, size=int(self.train_percent*l), p=p, replace=self.replace)
        if not self.repeat:
            train_indices = numpy.unique(train_indices)
        test_indices = numpy.ones(l, dtype=bool)
        test_indices[train_indices] = False
        return train_indices, test_indices

    def _pdf(self, x, mean):
        return exp(-squared_distance(x, mean) / (2 * self.variance)).real

    def _get_means_and_p(self, inp, n_means, random_state):
        mean_prob = numpy.ones(inp.shape[0]) / inp.shape[0]
        p = numpy.zeros(inp.shape[0])
        for i in range(n_means):
            mean = inp[random_state.choice(len(inp), p=mean_prob)]
            for j, x in enumerate(inp):
                p[j] = self._pdf(x, mean)
                mean_prob[j] *= (1 - p[j])
            mean_prob = mean_prob / mean_prob.sum()
            p = p / p.sum()
            yield mean, p
