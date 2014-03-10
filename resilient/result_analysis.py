import sys
import cPickle

import numpy
from sklearn.utils.validation import array2d

from resilient.logger import Logger


__author__ = 'tamponi'


if __name__ == '__main__':
    params = numpy.array([])
    scores = []
    for file_name in sys.argv[1:]:
        with open(file_name) as f:
            data = cPickle.load(f)
            if len(params) > 0 and numpy.any(data["params"] != params):
                print "Error: you are trying to load together incompatible results:", file_name
                exit(1)
            else:
                params = data["params"]
            scores.append(data["scores"].mean(axis=0))
    scores = array2d(scores)
    for i, row in enumerate(numpy.transpose(scores)):
        Logger.get().write("{}: {} - Mean: {:.3f}".format(numpy.array(params[i].values()), row, row.mean()))
    mean_scores = scores.mean(axis=0)
    best_mean_index = mean_scores.argmax()
    best_mean_param = params[best_mean_index]
    best_mean_score = mean_scores[best_mean_index]
    Logger.get().write("Best result:", best_mean_score, "with param:", best_mean_param)
