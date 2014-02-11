from datetime import datetime
from itertools import product
import sys

import arff
import numpy
from sklearn import cross_validation, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.selection_strategies import SelectBestK
from resilient.weighting_strategies import CentroidBasedWeightingStrategy
from resilient.splitting_strategies import CentroidBasedPDFSplittingStrategy, CentroidBasedKNNSplittingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


SEED = 1
N_ITER = 1
CV_METHOD = lambda t: cross_validation.StratifiedShuffleSplit(t, n_iter=N_ITER, test_size=0.1, random_state=SEED)
#CV_METHOD = lambda target: cross_validation.StratifiedKFold(target, n_folds=N_ITER)
N_ESTIMATORS = 1501
USE_WEIGHTS = False
MAX_FEATURES = "log2"

TRAINING_RANGE = numpy.linspace(0.10, 0.30, num=5)
# For BestK
SELECTION_STRATEGY = SelectBestK
TESTING_RANGE = range(1, N_ESTIMATORS+1, 2)


class Logger(object):
    def __init__(self, filename):
        numpy.set_printoptions(precision=3)
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.disable_log = False

    def write(self, message):
        self.terminal.write(message)
        if message.startswith("\r"):
            self.disable_log = True
        if not self.disable_log:
            self.log.write(message)
        if message.endswith("\n") and self.disable_log:
            self.disable_log = False
        self.flush()

    def finish(self):
        sys.stdout = self.terminal

    def flush(self):
        self.log.flush()
        self.terminal.flush()


def main():
    sys.stdout = Logger("results/experiment-humvar_01-{:%Y%m%d-%H%M%S}.txt".format(datetime.utcnow()))

    print "Experiment parameters"
    print "SEED:", SEED
    print "CV METHOD:", CV_METHOD([0, 1]*(N_ITER*10))
    print "N ESTIMATORS:", N_ESTIMATORS
    print "USE WEIGHTS:", USE_WEIGHTS
    print "MAX FEATURES:", MAX_FEATURES
    print "SELECTION STRATEGY:", SELECTION_STRATEGY.__name__

    data = []
    target = []
    for row in arff.load("humvar_10fold/humvar_01.arff"):
        row = list(row)
        target.append(row[-1])
        data.append(row[:-1])
    data = numpy.array(data)
    target = numpy.array(target)
    classes_, target = unique(target, return_inverse=True)

    #data = PCA(n_components=10, whiten=True).fit_transform(data)

    print "Running Random Forest..."
    cv = CV_METHOD(target)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, random_state=SEED)
    score = cross_validation.cross_val_score(rf, data, target, cv=cv)
    print "  RF  : {} - Mean: {:.3f}".format(score, score.mean())

    data = preprocessing.MinMaxScaler().fit_transform(data)

    for train_percent in TRAINING_RANGE:
        print "---------"
        print "Train   :", train_percent
        print "---------"
        ens = ResilientEnsemble(
            training_strategy=TrainingStrategy(
                n_estimators=N_ESTIMATORS,
                splitting_strategy=CentroidBasedKNNSplittingStrategy(
                    train_percent=train_percent
                ),
                base_estimator=DecisionTreeClassifier(max_features=MAX_FEATURES)
            ),
            weighting_strategy=CentroidBasedWeightingStrategy(),
            random_state=SEED,
            multiply_by_weight=USE_WEIGHTS
        )

        cv = CV_METHOD(target)
        results = {}
        i = -1
        for train_indices, test_indices in cv:
            i += 1
            print "\rRunning", (i+1), "of", N_ITER, "iterations..."
            train_data, train_target = data[train_indices], target[train_indices]
            test_data, test_target = data[test_indices], target[test_indices]
            ens.fit(train_data, train_target)
            for param in TESTING_RANGE:
                ens.set_params(selection_strategy=SELECTION_STRATEGY(param))
                if param not in results:
                    results[param] = numpy.zeros(N_ITER)
                results[param][i] = ens.score(test_data, test_target, scoring="accuracy")
                print "\rTesting using param:", param,
            print ""
        for param in TESTING_RANGE:
            if isinstance(param, int):
                print "{:5d}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())
            else:
                print "{:.3f}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())
        best_results = numpy.zeros(N_ITER)
        for i in range(N_ITER):
            result = numpy.array([results[param][i] for param in TESTING_RANGE])
            best_results[i] = result.max()
        print "       Best of means: {:.3f}".format(numpy.array([results[p].mean() for p in TESTING_RANGE]).max())
        print "       Mean of bests: {:.3f}".format(best_results.mean())


if __name__ == '__main__':
    main()
