import cmath

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import pdfs, weighting_strategies
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.train_set_generators import RandomCentroidPDFTrainSetGenerator


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


_centroids = lambda inner, outer: ResilientEnsemble(
    preprocessing_pipeline=Pipeline(
        steps=[
            ("scale", MinMaxScaler())
        ]
    ),
    n_estimators=outer,
    training_strategy=TrainingStrategy(
        base_estimator=RandomForestClassifier(
            bootstrap=False,
            n_estimators=inner,
            max_features=4,
            criterion="entropy"
        ),
        train_set_generator=RandomCentroidPDFTrainSetGenerator(
            pdf=pdfs.DistanceExponential(
                tau=0.10,
                base=cmath.e,
                dist_measure="euclidean"
            )
        )
    ),
    weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(
        dist_measure="euclidean"
    ),
    multiply_by_weight=False,
    use_prob=True
)


example_centroids = _centroids(5, 5)
centroids_20_50 = _centroids(20, 50)
