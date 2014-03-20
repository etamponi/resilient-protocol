import sys
import cPickle

from matplotlib import pyplot
import numpy
from sklearn.utils.validation import array2d

from resilient.logger import Logger


__author__ = 'tamponi'


if __name__ == '__main__':
    params = numpy.array([])
    scores = []
    all_scores = []
    for file_name in sys.argv[1:]:
        with open(file_name) as f:
            data = cPickle.load(f)
            params = [p.values()[0] for p in data["params"]]
            scores.append(data["scores"].mean(axis=0))
            for s in data["scores"]:
                all_scores.append(s)
    scores = array2d(scores)
    all_scores = array2d(all_scores)
    for i, row in enumerate(numpy.transpose(scores)):
        Logger.get().write("{}: {} - Mean: {:.4f}".format(numpy.array([params[i]]), row, row.mean()))
    mean_scores = scores.mean(axis=0)
    best_mean_index = mean_scores.argmax()
    best_mean_param = params[best_mean_index]
    best_mean_score_row = all_scores[:, best_mean_index]
    best_mean_score = mean_scores[best_mean_index]
    best_mean_score_stddev = best_mean_score_row.std()
    Logger.get().write("Best result: {:.4f} +- {:.4f} with param {:.3f}".format(
        best_mean_score, best_mean_score_stddev, best_mean_param
    ))
    pyplot.plot(params, mean_scores)
    pyplot.show()
