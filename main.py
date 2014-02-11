import arff
import numpy
import scipy.spatial.distance
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier
from resilient import experiment
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient.pdfs import MultivariateExponential
from resilient.splitting_strategies import CentroidBasedPDFSplittingStrategy
from resilient.weighting_strategies import CentroidBasedWeightingStrategy

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


SEED = 1
N_ITER = 2
CV_METHOD = lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s)


def main():
    data = []
    target = []
    for row in arff.load("humvar_10fold/humvar_01.arff"):
        row = list(row)
        target.append(row[-1])
        data.append(row[:-1])
    data = numpy.array(data)
    target = numpy.array(target)

    pipeline = Pipeline(
        steps=[
            ("scale", MinMaxScaler())
        ]
    )

    experiment.run_experiment(
        dataset_name="humvar_01",
        data=data,
        target=target,
        preprocessing_pipeline=pipeline,
        ensemble=ResilientEnsemble(
            training_strategy=TrainingStrategy(
                n_estimators=301,
                base_estimator=DecisionTreeClassifier(
                    max_features=4,
                    max_depth=20
                ),
                splitting_strategy=CentroidBasedPDFSplittingStrategy(
                    pdf=MultivariateExponential(
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
        cv_method=CV_METHOD,
        n_iter=N_ITER,
        seed=SEED,
        log_filename="test.txt"
    )


if __name__ == "__main__":
    main()
