"""
Standard :class:`resilient.selection_strategies.SelectionStrategy` objects.

This module contains the standard selection strategies that can be used together
with :mod:`experiment_launcher`.
"""

from resilient import selection_strategies

__author__ = 'tamponi'


select_best = selection_strategies.SelectBestPercent()

select_weight_sum = selection_strategies.SelectByWeightSum()

select_threshold = selection_strategies.SelectByWeightThreshold()

select_random = selection_strategies.SelectRandomPercent()

select_static = selection_strategies.StaticSelection()
