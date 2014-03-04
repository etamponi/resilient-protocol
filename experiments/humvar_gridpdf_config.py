import arff
import numpy
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn import preprocessing

from resilient import pdfs, selection_strategies
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.train_set_generators import GridPDFTrainSetGenerator
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'

x = 1
with open("../humvar_10fold/humvar_{:02d}.arff".format(x)) as f:
    d = arff.load(f)
    data = numpy.array([row[:-1] for row in d['data']])
    target = numpy.array([row[-1] for row in d['data']])

    data = preprocessing.MinMaxScaler().fit_transform(data)


precision = EmpiricalCovariance(store_precision=True, assume_centered=False).fit(data).get_precision()


def mahalanobis_distance(a, b):
    return distance.mahalanobis(a, b, VI=precision)


config = {
    "seed": 1,
    "n_iter": 10,
    #"cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s),
    "cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    #"cv_method": lambda t, n, s: cross_validation.KFold(len(t), n_folds=n, random_state=s),
    "dataset_name": "humvar_{:02d}".format(x),
    "data": data,
    "target": target,
    # "pipeline": Pipeline(
    #     steps=[
    #         ("scale", preprocessing.MinMaxScaler())
    #     ]
    # ),
    "pipeline": None,
    "ensemble": ResilientEnsemble(
        training_strategy=TrainingStrategy(
            base_estimator=RandomForestClassifier(
                n_estimators=21,
                max_features=4,
                criterion="entropy",
                bootstrap=False,
                max_depth=20
            ),
            train_set_generator=GridPDFTrainSetGenerator(
                n_estimators=81,
                spacing=0.5,
                pdf=pdfs.DistanceExponential(
                    tau=0.25,
                    dist_measure=mahalanobis_distance
                ),
                percent=2.0,
                replace=True,
                repeat=True
            )
        ),
        weighting_strategy=CentroidBasedWeightingStrategy(
            dist_measure=mahalanobis_distance
        ),
        multiply_by_weight=False,
        use_prob=True,
        validation_percent=0.05
    ),
    "selection_strategy": selection_strategies.SelectByWeightSum(
        param=0.10,
        kernel=numpy.ones(5)/5
    ),
    "rf": None,
    "use_mcc": False
}


if __name__ == "__main__":
    from resilient import experiment as exp
    exp.run_experiment(**config)
