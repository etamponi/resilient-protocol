import cmath

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from resilient import train_set_generators, pdfs, weighting_strategies
from resilient.ensemble import ResilientEnsemble, TrainingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


def generalized_exponential_ensemble(n_estimators, inner_estimators, precision, base, power, max_features,
                                     random_sample=None, criterion="entropy", multiply_by_weight=False):
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
            train_set_generator=train_set_generators.RandomCentroidPDFTrainSetGenerator(
                pdf=pdfs.DistanceGeneralizedExponential(precision=precision, base=base, power=power)
            ),
            random_sample=random_sample
        ),
        weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(),
        multiply_by_weight=multiply_by_weight,
        use_prob=True
    )


def _ensemble_gen(n_estimators, inner_estimators, random_sample, criterion, max_features, pdf, weighting, mul, prob):
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
        multiply_by_weight=mul,
        use_prob=prob
    )


small_centroids_100 = _ensemble_gen(
    100, 1, None, "entropy", "auto", pdfs.DistanceExponential(tau=0.01, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
small_centroids_10 = _ensemble_gen(
    10, 1, None, "gini", "auto", pdfs.DistanceExponential(tau=0.02, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

centroids_50_20 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
centroids_1000_10 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
centroids_1000_05 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceExponential(tau=0.05, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
centroids_1000_03 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceExponential(tau=0.03, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
centroids_100_10_05 = _ensemble_gen(
    100, 10, None, "entropy", 4, pdfs.DistanceExponential(tau=0.05, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
centroids_200_05_05 = _ensemble_gen(
    200, 5, None, "entropy", 4, pdfs.DistanceExponential(tau=0.05, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

random_forest_1000 = _ensemble_gen(
    1000, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.SameWeight(), False, False
)
random_forest_1000_sorted = _ensemble_gen(
    1000, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.CentroidBasedWeightingStrategy(), False, False
)
random_forest_1000_prob = _ensemble_gen(
    1000, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.SameWeight(), False, True
)
random_forest_50_20 = _ensemble_gen(
    50, 20, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.SameWeight(), False, False
)
random_forest_100 = _ensemble_gen(
    100, 1, 1.0, "gini", 4, pdfs.Uniform(), weighting_strategies.SameWeight(), False, False
)

base_2_50_20_10 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceExponential(tau=0.10, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

inverse_50_20_02 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceInverse(power=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
inverse_50_20_10 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceInverse(power=10),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

normal_2_50_20_100 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceNormal(precision=100, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_50_20_50 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceNormal(precision=50, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_e_50_20_50 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceNormal(precision=50, base=cmath.e),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_50_20_30 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistanceNormal(precision=30, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_100_10_30 = _ensemble_gen(
    100, 10, None, "entropy", 4, pdfs.DistanceNormal(precision=30, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_200_05_30 = _ensemble_gen(
    200, 5, None, "entropy", 4, pdfs.DistanceNormal(precision=30, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_1000_1_60 = _ensemble_gen(
    1000, 1, None, "entropy", 4, pdfs.DistanceNormal(precision=60, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

normal_2_100_1_50 = _ensemble_gen(
    100, 1, None, "entropy", 4, pdfs.DistanceNormal(precision=50, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_100_1_40 = _ensemble_gen(
    100, 1, None, "entropy", 4, pdfs.DistanceNormal(precision=40, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
normal_2_100_1_30 = _ensemble_gen(
    100, 1, None, "entropy", 4, pdfs.DistanceNormal(precision=30, base=2),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

genexp_100_1_30_2_15 = _ensemble_gen(
    100, 1, None, "entropy", 4, pdfs.DistanceGeneralizedExponential(precision=30, base=2, power=1.5),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)

power_50_20_05 = _ensemble_gen(
    50, 20, None, "entropy", 4, pdfs.DistancePower(power=0.5),
    weighting_strategies.CentroidBasedWeightingStrategy(), False, True
)
