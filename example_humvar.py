from itertools import product
import sys

import numpy
import pandas
from sklearn import cross_validation, preprocessing
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.selection_strategies import SelectBestK
from resilient.sorting_strategies import CentroidBasedWeightingStrategy
from resilient.splitting_strategies import CentroidBasedSplittingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


SEED = 5
N_ITER = 5
CV_METHOD = lambda target: cross_validation.ShuffleSplit(len(target), n_iter=N_ITER, test_size=0.1, random_state=SEED)

N_ESTIMATORS = 201
VARIANCE_RANGE = numpy.linspace(0.05, 0.15, num=3)
TRAINING_RANGE = numpy.linspace(0.25, 0.35, num=3)
# For BestK
SELECTION_STRATEGY = SelectBestK
TESTING_RANGE = range(1, N_ESTIMATORS+1, N_ESTIMATORS/20)


class Logger(object):
    def __init__(self, filename):
        numpy.set_printoptions(precision=3)
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def finish(self):
        sys.stdout = self.terminal

    def flush(self):
        self.log.flush()
        self.terminal.flush()


def main():
    sys.stdout = Logger('experiments_humvar.txt')

    with open('humvar.csv') as f:
        data = pandas.read_csv(f)
        features = data.columns[:-1]
        data, target = data[features].values, data["dataset"].values
        classes_, target = unique(target, return_inverse=True)

    #data = PCA(n_components=10).fit_transform(data)
    data = preprocessing.MinMaxScaler().fit_transform(data)

    print "Running Random Forest..."
    cv = CV_METHOD(target)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features=4, random_state=SEED)
    score = cross_validation.cross_val_score(rf, data, target, cv=cv)
    print "  RF  : {} - Mean: {:3f}".format(score, score.mean())

    for variance, train_percent in product(VARIANCE_RANGE, TRAINING_RANGE):
        print "Variance:", variance
        print "Train   :", train_percent
        ens = ResilientEnsemble(
            training_strategy=TrainingStrategy(
                n_estimators=N_ESTIMATORS,
                splitting_strategy=CentroidBasedSplittingStrategy(
                    variance=variance,
                    train_percent=train_percent,
                    replace=True,
                    repeat=True
                ),
                base_estimator=DecisionTreeClassifier(max_features=4)
            ),
            weighting_strategy=CentroidBasedWeightingStrategy(),
            random_state=SEED
        )

        cv = CV_METHOD(target)
        results = {}
        i = -1
        for train_indices, test_indices in cv:
            i += 1
            print "Running", (i+1), "of", N_ITER, "iterations."
            train_data, train_target = data[train_indices], target[train_indices]
            test_data, test_target = data[test_indices], target[test_indices]
            ens.fit(train_data, train_target)
            for param in TESTING_RANGE:
                ens.set_params(selection_strategy=SELECTION_STRATEGY(param))
                if param not in results:
                    results[param] = numpy.zeros(N_ITER)
                results[param][i] = ens.score(test_data, test_target)
                print "\rTesting using param:", param,
            print ""
        for param in TESTING_RANGE:
            if isinstance(param, int):
                print "{:5d}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())
            else:
                print "{:.3f}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())


if __name__ == '__main__':
    main()
