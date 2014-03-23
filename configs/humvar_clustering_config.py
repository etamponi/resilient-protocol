import sys

import arff
import numpy
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import pdfs, selection_strategies, selection_optimizers, weighting_strategies, train_set_generators
from resilient.ensemble import ResilientEnsemble, TrainingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


def get_config():
    x, results_dir = int(sys.argv[1]), sys.argv[2]

    with open("../humvar_10fold/humvar_{:02d}.arff".format(x)) as f:
        d = arff.load(f)
        data = numpy.array([row[:-1] for row in d['data']])
        target = numpy.array([row[-1] for row in d['data']])

    config = {
        "seed": 1,
        "n_iter": 10,
        #"cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n, test_size=0.1, random_state=s),
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
                    n_estimators=200,
                    max_features=4,
                    criterion="entropy"
                ),
                train_set_generator=train_set_generators.ClusteringPDFTrainSetGenerator(
                    clustering=train_set_generators.KMeansWrapper(
                        n_estimators=5,
                        use_mini_batch=True,
                        max_iter=100
                    ),
                    pdf=pdfs.DistanceExponential(
                        tau=0.25,
                        base=2
                    )
                )
            ),
            selection_strategy=selection_strategies.SelectBestPercent(
                percent=0.60
            ),
            selection_optimizer=selection_optimizers.GridOptimizer(
                kernel_size=5,
                custom_ranges={
                    "percent": numpy.linspace(0, 1, 11)[1:],
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
        "results_dir": results_dir,
        "run_async": False
    }
    return config


if __name__ == "__main__":
    from resilient import experiment as exp
    exp.run_cv_training(**get_config())
