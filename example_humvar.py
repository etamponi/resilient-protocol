from datetime import datetime
from itertools import product
import sys

import arff
import numpy
from sklearn import cross_validation, preprocessing
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import unique

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.logger import Logger
from resilient.selection_strategies import SelectBestK
from resilient.weighting_strategies import CentroidBasedWeightingStrategy
from resilient.splitting_strategies import CentroidBasedPDFSplittingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


SEED = 1
N_ITER = 2
CV_METHOD = lambda t: cross_validation.StratifiedShuffleSplit(t, n_iter=N_ITER, test_size=0.1, random_state=SEED)
#CV_METHOD = lambda t: cross_validation.StratifiedKFold(t, n_folds=N_ITER)
N_ESTIMATORS = 11
REPLACE = False
REPEAT = False
USE_WEIGHTS = False
USE_PROB = True
MAX_FEATURES = 4

MAX_DEPTH_RANGE = range(10, 41, 10)
VARIANCE_RANGE = numpy.linspace(0.15, 0.25, num=1)
TRAINING_RANGE = numpy.linspace(0.35, 0.40, num=1)
# For BestK
SELECTION_STRATEGY = SelectBestK
TESTING_RANGE = range(1, N_ESTIMATORS+1, 2)


def main():
    sys.stdout = Logger("results/experiment-humvar_01-{:%Y%m%d-%H%M%S}.txt".format(datetime.utcnow()))

    print "Experiment parameters"
    print "SEED:", SEED
    print "CV METHOD:", CV_METHOD([0, 1]*(N_ITER*10))
    print "N ESTIMATORS:", N_ESTIMATORS
    print "REPLACE:", REPLACE
    print "REPEAT:", REPEAT
    print "USE WEIGHTS:", USE_WEIGHTS
    print "USE PROB:", USE_PROB
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

    #data = PCA().fit_transform(data)

    print "Running Random Forest..."
    cv = CV_METHOD(target)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, random_state=SEED)
    score = cross_validation.cross_val_score(rf, data, target, cv=cv)
    print "  RF  : {} - Mean: {:.3f}".format(score, score.mean())

    data = preprocessing.MinMaxScaler().fit_transform(data)

    for variance, train_percent, max_depth in product(VARIANCE_RANGE, TRAINING_RANGE, MAX_DEPTH_RANGE):
        print "--------------------------"
        print "Variance :", variance
        print "Train    :", train_percent
        print "Max depth:", max_depth
        print "--------------------------"
        ens = ResilientEnsemble(
            training_strategy=TrainingStrategy(
                n_estimators=N_ESTIMATORS,
                splitting_strategy=CentroidBasedPDFSplittingStrategy(
                    pdf_params=variance,
                    train_percent=train_percent,
                    replace=REPLACE,
                    repeat=REPEAT
                ),
                base_estimator=DecisionTreeClassifier(max_features=MAX_FEATURES, max_depth=max_depth)
            ),
            weighting_strategy=CentroidBasedWeightingStrategy(),
            multiply_by_weight=USE_WEIGHTS,
            use_prob=USE_PROB,
            random_state=SEED
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
                print "{:6d}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())
            else:
                print "{:6.3f}: {} - Mean: {:.3f}".format(param, results[param], results[param].mean())
        best_results = numpy.zeros(N_ITER)
        best_params = numpy.zeros(N_ITER, dtype=int)
        for i in range(N_ITER):
            result = numpy.array([results[param][i] for param in TESTING_RANGE])
            best_results[i] = result.max()
            best_params[i] = TESTING_RANGE[result.argmax()]
        print "Best p: {} - Mean p of bst: {:d}".format(best_params, int(best_params.mean()))
        print "  Best: {} - Mean of bests: {:.3f}".format(best_results, best_results.mean())
        mean_results = numpy.array([results[p].mean() for p in TESTING_RANGE])
        best_mean_param = TESTING_RANGE[mean_results.argmax()]
        best_mean_r = results[best_mean_param]
        print "Best m: {} - Best of means: {:.3f} (param {:d})".format(best_mean_r, best_mean_r.mean(), best_mean_param)


if __name__ == '__main__':
    main()
