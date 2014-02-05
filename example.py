import sys

import numpy
from sklearn import cross_validation, preprocessing
from sklearn.datasets.samples_generator import make_hastie_10_2
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.utils.random import check_random_state

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.selection_strategies import SelectBestK
from resilient.sorting_strategies import CentroidBasedSortingStrategy
from resilient.splitting_strategies import CentroidBasedSplittingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


N_ESTIMATORS = 99
N_ITER = 5


def main():
    random_state = check_random_state(10)

    data, target = make_hastie_10_2(n_samples=1000, random_state=random_state)
    data = preprocessing.MinMaxScaler().fit_transform(data)
    ens = ResilientEnsemble(
        training_strategy=TrainingStrategy(
            n_estimators=N_ESTIMATORS,
            splitting_strategy=CentroidBasedSplittingStrategy(
                variance=0.2,
                train_percent=0.6,
                replace=True
            )
        ),
        sorting_strategy=CentroidBasedSortingStrategy(),
        random_state=random_state
    )

    cv = cross_validation.StratifiedShuffleSplit(target, n_iter=5, test_size=0.20, random_state=random_state)
    results = {}
    i = -1
    for train_indices, test_indices in cv:
        i += 1
        sys.stdout.flush()
        train_data, train_target = data[train_indices], target[train_indices]
        test_data, test_target = data[test_indices], target[test_indices]
        ens.fit(train_data, train_target)
        for k in range(1, N_ESTIMATORS+1, 2):
        #for k in linspace(0.5, 0.8, 5):
            print k,
            ens.set_params(selection_strategy=SelectBestK(k))
            if k not in results:
                results[k] = numpy.zeros(cv.n_iter)
            results[k][i] = ens.score(test_data, test_target)
        print ""
    for k in sorted(results.keys()):
        print k, results[k], results[k].mean()

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=random_state)
    score = cross_validation.cross_val_score(rf, data, target, cv=cv)
    print score, score.mean()


if __name__ == '__main__':
    main()
