from resilient import selection_strategies

__author__ = 'tamponi'


best = selection_strategies.SelectBestPercent()
weight_sum = selection_strategies.SelectByWeightSum()
threshold = selection_strategies.SelectByWeightThreshold()
