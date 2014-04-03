from sklearn import preprocessing
from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.pipeline import Pipeline

from resilient import train_set_generators, pdfs, weighting_strategies

from resilient.ensemble import ResilientEnsemble, TrainingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


def generalized_exponential_ensemble(n_estimators, inner_estimators, precision, power, max_features,
                                     scaler="minmax", weighting_power=None, random_sample=None, criterion="entropy"):
    if weighting_power is None:
        weighting_power = 1
        multiply_by_weight = False
    else:
        multiply_by_weight = True
    if scaler == "minmax":
        scaler_obj = preprocessing.MinMaxScaler()
    elif scaler == "standard":
        scaler_obj = preprocessing.StandardScaler()
    else:
        raise Exception("Scaler must be \"minmax\" or \"standard\".")

    return ResilientEnsemble(
        pipeline=Pipeline(
            steps=[
                (scaler, scaler_obj)
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
                pdf=pdfs.DistanceGeneralizedExponential(precision=precision, power=power)
            ),
            random_sample=random_sample
        ),
        weighting_strategy=weighting_strategies.CentroidBasedWeightingStrategy(
            pdf=pdfs.DistanceInverse(power=weighting_power)
        ),
        multiply_by_weight=multiply_by_weight,
        use_prob=True
    )
