from resilient.cross_validation import NestedStratifiedKFold

__author__ = 'tamponi'


example_nested = NestedStratifiedKFold(n_runs=2, n_folds=2, seed=1)
