"""
Standard cross-validation objects to be used with :mod:`experiment_launcher`

The default cross-validation object is ``not_nested_10fold``, which stands for
a default 10-fold stratified cross-validation ran only once. Check the
documentation for :class:`NestedStratifiedKFold` for further details on how
the nested cross-validation is done.

There exists also a ``example_nested`` to be run for testing purposes.
"""

from resilient.cross_validation import NestedStratifiedKFold

__author__ = 'tamponi'


example_nested = NestedStratifiedKFold(n_runs=2, n_folds=2, seed=1)

not_nested_10fold = NestedStratifiedKFold(n_runs=1, n_folds=10, seed=1)
