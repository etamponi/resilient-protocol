from sklearn import preprocessing
from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.pipeline import Pipeline

from resilient import train_set_generators, pdfs, weighting_strategies

from resilient.ensemble import ResilientEnsemble, TrainingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


minmax_pipeline = Pipeline(
    steps=[("minmax", preprocessing.MinMaxScaler())]
)

standard_pipeline = Pipeline(
    steps=[("standard", preprocessing.StandardScaler())]
)


def generalized_exponential_resilient_forest(
        n_estimators, inner_estimators, max_features, precision, power,
        pipeline=minmax_pipeline, weighting_power=None, random_sample=None,
        criterion="entropy", max_depth=None):
    if weighting_power is None:
        weighting_power = 1
        multiply_by_weight = False
    else:
        multiply_by_weight = True

    return ResilientEnsemble(
        pipeline=pipeline,
        n_estimators=n_estimators,
        training_strategy=TrainingStrategy(
            base_estimator=RandomForestClassifier(
                bootstrap=False,
                n_estimators=inner_estimators,
                max_features=max_features,
                criterion=criterion,
                max_depth=max_depth
            ),
            train_set_generator=
            train_set_generators.RandomCentroidPDFTrainSetGenerator(
                pdf=pdfs.DistanceGeneralizedExponential(precision=precision,
                                                        power=power)
            ),
            random_sample=random_sample
        ),
        weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(
            pdf=pdfs.DistanceInverse(power=weighting_power)
        ),
        multiply_by_weight=multiply_by_weight,
        use_prob=True
    )
