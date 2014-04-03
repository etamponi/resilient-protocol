from resilient.cross_validation import NestedStratifiedKFold

__author__ = 'tamponi'


example_nested = NestedStratifiedKFold(n_runs=2, n_folds=2, seed=1)
not_nested_10fold = NestedStratifiedKFold(n_runs=1, n_folds=10, seed=1)