from datetime import datetime

import arff
import numpy
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import pdfs
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.splitting_strategies import CentroidBasedPDFSplittingStrategy
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

x = 1

with open("humvar_10fold/humvar_{:02d}.arff".format(x)) as f:
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
                n_estimators=51,
                max_features=4,
                criterion="entropy",
                bootstrap=False,
                max_depth=20
            ),
            splitting_strategy=CentroidBasedPDFSplittingStrategy(
                n_estimators=71,
                pdf=pdfs.DistanceExponential(
                    tau=0.25,
                    dist_measure=distance.euclidean
                ),
                train_percent=2.5,
                replace=True,
                repeat=True
            )
        ),
        weighting_strategy=CentroidBasedWeightingStrategy(
            dist_measure=distance.euclidean
        ),
        multiply_by_weight=False,
        use_prob=True
    ),
    "log_filename": "results/experiment-humvar-centroid-{:02d}-{:%Y%m%d-%H%M-%S}.txt".format(x, datetime.utcnow()),
    "rf_trees": None
}
