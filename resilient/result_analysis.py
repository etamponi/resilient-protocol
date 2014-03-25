import numpy


__author__ = 'tamponi'


def confusion_to_accuracy(cm):
    correct = cm.diagonal().sum()
    total = cm.sum()
    return correct / total


def results_to_scores(results, confusion_to_score):
    iterations = results.shape[0]
    trials = results.shape[1]
    scores = numpy.zeros((iterations, trials))
    for it in xrange(iterations):
        for t in xrange(trials):
            scores[it, t] = confusion_to_score(results[it, t])
    return scores
