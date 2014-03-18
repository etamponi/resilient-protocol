import sys

import arff
import cmath
import numpy
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import pdfs, selection_strategies, selection_optimizers, weighting_strategies
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.train_set_generators import RandomCentroidPDFTrainSetGenerator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


if len(sys.argv) > 1:
    x = int(sys.argv[1])
else:
    x = 1

with open("../humvar_10fold/humvar_{:02d}.arff".format(x)) as f:
    d = arff.load(f)
    data = numpy.array([row[:-1] for row in d['data']])
    target = numpy.array([row[-1] for row in d['data']])


config = {
    "seed": 1,
    "n_iter": 10,
    #"cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s),
    "cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    #"cv_method": lambda t, n, s: cross_validation.KFold(len(t), n_folds=n, random_state=s),
    "dataset_name": "humvar_{:02d}".format(x),
    "data": data,
    "target": target,
    "pipeline": Pipeline(
        steps=[
            ("scale", MinMaxScaler())
        ]
    ),
    # "pipeline": None,
    "ensemble": ResilientEnsemble(
        training_strategy=TrainingStrategy(
            base_estimator=RandomForestClassifier(
                bootstrap=False,
                n_estimators=25,
                max_features=4,
                criterion="entropy"
            ),
            train_set_generator=RandomCentroidPDFTrainSetGenerator(
                n_estimators=100,
                pdf=pdfs.DistanceNormal(
                    precision=5,
                    base=cmath.e
                )
            )
        ),
        selection_strategy=selection_strategies.SelectBestPercent(
            percent=0.60
        ),
        selection_optimizer=selection_optimizers.GridOptimizer(
            kernel_size=5,
            custom_ranges={
                "percent": numpy.linspace(0, 1, 101)[1:],
                "threshold": numpy.linspace(0, 1, 101)[1:]
            }
        ),
        weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(
            dist_measure=distance.euclidean
        ),
        multiply_by_weight=False,
        use_prob=True,
        validation_percent=None
    ),
    "rf": None,
    "use_mcc": False,
    "results_dir": "results_20140313_13"
}


if __name__ == "__main__":
    from resilient import experiment as exp
    exp.run_experiment(**config)
