import datetime

import arff
import numpy
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient import splitting_strategies
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


with open("humvar_10fold/humvar_01.arff") as f:
    d = arff.load(f)
    data = numpy.array([row[:-1] for row in d['data']])
    target = numpy.array([row[-1] for row in d['data']])


config = {
    "seed": 1,
    "n_iter": 10,
    #"cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s),
    "cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    "dataset_name": "humvar_01",
    "data": data,
    "target": target,
    # "pipeline": Pipeline(
    #     steps=[
    #         ("scale", MinMaxScaler(feature_range=(0, 5)))
    #     ]
    # ),
    "pipeline": None,
    "ensemble": ResilientEnsemble(
        training_strategy=TrainingStrategy(
            base_estimator=DecisionTreeClassifier(
                max_features=4,
                max_depth=25
            ),
            splitting_strategy=splitting_strategies.GridSplittingStrategy(
                n_cells_per_dim=5,
                min_cell_size=3,
                k_overlap=60,
                dist_measure=distance.cityblock
            )
        ),
        weighting_strategy=CentroidBasedWeightingStrategy(
            dist_measure=distance.euclidean
        ),
        multiply_by_weight=False,
        use_prob=True
    ),
    "log_filename": "results/experiment-humvar-grid-{:%Y%m%d-%H%M-%S}.txt".format(datetime.datetime.utcnow()),
    "rf_trees": None
}
