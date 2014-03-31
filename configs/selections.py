from resilient import selection_strategies

__author__ = 'tamponi'


best = selection_strategies.SelectBestPercent()
weight_sum = selection_strategies.SelectByWeightSum()
threshold = selection_strategies.SelectByWeightThreshold()
exp_random = selection_strategies.SelectRandomPercent()

skipping_2 = selection_strategies.SelectBestPercentSkipping(step=2)
skipping_3 = selection_strategies.SelectBestPercentSkipping(step=3)
skipping_4 = selection_strategies.SelectBestPercentSkipping(step=4)
