import cmath
import numpy
from sklearn import cross_validation

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import pdfs, selection_strategies, weighting_strategies

from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.train_set_generators import RandomCentroidPDFTrainSetGenerator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


N_INNER_ESTIMATORS = 20
N_ESTIMATORS = 50
config = {
    "seed": 1,
    "n_iter": 10,
    #"cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n, test_size=0.1, random_state=s),
    "cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    "ensemble": ResilientEnsemble(
        preprocessing_pipeline=Pipeline(
            steps=[
                ("scale", MinMaxScaler())
            ]
        ),
        training_strategy=TrainingStrategy(
            base_estimator=RandomForestClassifier(
                bootstrap=False,
                n_estimators=N_INNER_ESTIMATORS,
                max_features=4,
                criterion="entropy"
            ),
            train_set_generator=RandomCentroidPDFTrainSetGenerator(
                n_estimators=N_ESTIMATORS,
                pdf=pdfs.DistanceExponential(
                    tau=0.10,
                    base=cmath.e,
                    dist_measure="euclidean"
                )
            )
        ),
        selection_strategy=selection_strategies.SelectBestPercent(
            threshold=0.60
        ),
        weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(
            dist_measure="euclidean"
        ),
        multiply_by_weight=False,
        use_prob=True
    ),
    "threshold_range": numpy.linspace(0, 1, num=N_ESTIMATORS+1)[1:],
    "run_async": True
}
