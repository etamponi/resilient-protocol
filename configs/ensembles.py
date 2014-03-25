import cmath
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import train_set_generators, pdfs, weighting_strategies

from resilient.ensemble import ResilientEnsemble, TrainingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


def _ensemble_gen(n_estimators, inner_estimators, random_sample, criterion, max_features, pdf, weighting, use_prob):
    return ResilientEnsemble(
        preprocessing_pipeline=Pipeline(
            steps=[
                ("scale", MinMaxScaler())
            ]
        ),
        n_estimators=n_estimators,
        training_strategy=TrainingStrategy(
            base_estimator=RandomForestClassifier(
                bootstrap=False,
                n_estimators=inner_estimators,
                max_features=max_features,
                criterion=criterion
            ),
            train_set_generator=train_set_generators.RandomCentroidPDFTrainSetGenerator(pdf=pdf),
            random_sample=random_sample
        ),
        weighting_strategy=weighting,
        multiply_by_weight=False,
        use_prob=use_prob
    )


example_centroids = _ensemble_gen(5, 5, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=cmath.e),
                                  weighting_strategies.CentroidBasedWeightingStrategy(), True)

centroids_50_20 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), True
)
centroids_1000_10 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), True
)
centroids_1000_05 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceExponential(tau=0.05, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), True
)

random_forest_1000 = _ensemble_gen(
    1000, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.SameWeight(), False
)
random_forest_1000_sorted = _ensemble_gen(
    1000, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.CentroidBasedWeightingStrategy(), False
)
