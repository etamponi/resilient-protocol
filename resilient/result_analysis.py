import cmath

import numpy


__author__ = 'tamponi'


def confusion_to_accuracy(cm):
    correct = cm.diagonal().sum()
    total = cm.sum()
    return correct / total


def confusion_to_matthews(cm):
    ((tn, fn), (fp, tp)) = cm
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0:
        return ((tp * tn) - (fp * fn)) / cmath.sqrt(denominator).real
    else:
        return 0


def results_to_scores(results, confusion_to_score=confusion_to_accuracy):
    iterations = results.shape[0]
    trials = results.shape[1]
    scores = numpy.zeros((iterations, trials))
    for it in xrange(iterations):
        for t in xrange(trials):
            scores[it, t] = confusion_to_score(results[it, t])
    return scores
