from abc import ABCMeta, abstractmethod
import numpy
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils.random import check_random_state

__author__ = 'tamponi'


class CrossValidation(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, y):
        """
        Like the cross validation of scikit-learn, but also return a seed
        """
        pass

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def total_runs(self):
        pass


class NestedStratifiedKFold(CrossValidation):

    def __init__(self, n_runs=10, n_folds=10, seed=1):
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.seed = seed

    def build(self, y):
        random_state = check_random_state(self.seed)
        for run in xrange(self.n_runs):
            shuffled_y = y[random_state.permutation(len(y))]
            for train_indices, test_indices in StratifiedKFold(shuffled_y, n_folds=self.n_folds):
                seed = random_state.randint(numpy.iinfo(numpy.int32).max)
                yield seed, train_indices, test_indices

    def get_filename(self):
        return "{name}_seed{seed:02d}_{runs:02d}_{folds:02d}".format(
            name=self.__class__.__name__,
            seed=self.seed,
            runs=self.n_runs,
            folds=self.n_folds
        )

    def total_runs(self):
        return self.n_runs * self.n_folds
