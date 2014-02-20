import arff
import datetime
import numpy
import scipy.spatial.distance
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from resilient import pdfs
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.splitting_strategies import CentroidBasedPDFSplittingStrategy
from resilient.weighting_strategies import CentroidBasedWeightingStrategy

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


with open("humvar_10fold/humvar_01.arff") as f:
    d = arff.load(f)
    data = numpy.array([row[:-1] for row in d['data']])
    target = numpy.array([row[-1] for row in d['data']])


config = {
    "seed": 1,
    "n_iter": 2,
    "cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s),
    #"cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    "dataset_name": "humvar_01",
    "data": data,
    "target": target,
    "pipeline": Pipeline(
        steps=[
            ("scale", MinMaxScaler())
        ]
    ),
    "ensemble": ResilientEnsemble(
        training_strategy=TrainingStrategy(
            base_estimator=DecisionTreeClassifier(
                max_features=4,
                max_depth=20
            ),
            splitting_strategy=CentroidBasedPDFSplittingStrategy(
                n_estimators=301,
                pdf=pdfs.DistanceExponential(
                    tau=0.15,
                    dist_measure=scipy.spatial.distance.euclidean
                ),
                train_percent=0.35,
                replace=False,
                repeat=False
            )
        ),
        weighting_strategy=CentroidBasedWeightingStrategy(
            dist_measure=scipy.spatial.distance.euclidean
        ),
        multiply_by_weight=False,
        use_prob=True
    ),
    "log_filename": "results/experiment-humvar-centroid-{:%Y%m%d-%H%M-%S}.txt".format(datetime.datetime.utcnow()),
    "rf_trees": None
}
