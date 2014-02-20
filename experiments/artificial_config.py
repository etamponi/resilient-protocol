import datetime

from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.datasets.samples_generator import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from resilient import pdfs
from resilient.ensemble import ResilientEnsemble, TrainingStrategy
from resilient import splitting_strategies
from resilient.weighting_strategies import CentroidBasedWeightingStrategy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


data, target = make_classification(
    n_samples=3000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=10,
    flip_y=0.05,
    class_sep=1.0,
    random_state=1
)


config = {
    "seed": 1,
    "n_iter": 2,
    "cv_method": lambda t, n, s: cross_validation.StratifiedShuffleSplit(t, n_iter=n, test_size=0.1, random_state=s),
    #"cv_method": lambda t, n, s: cross_validation.StratifiedKFold(t, n_folds=n),
    "dataset_name": "artificial",
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
                max_features="auto",
                max_depth=None
            ),
            splitting_strategy=splitting_strategies.GridSplittingStrategy(
                n_cells_per_dim=5
            )
        ),
        weighting_strategy=CentroidBasedWeightingStrategy(
            dist_measure=distance.euclidean
        ),
        multiply_by_weight=True,
        use_prob=True
    ),
    "log_filename": "results/experiment-{:%Y%m%d-%H%M-%S}.txt".format(datetime.datetime.utcnow()),
    "run_rf": True
}