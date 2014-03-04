from datetime import datetime
import sys

import numpy
from numpy.core.fromnumeric import transpose
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.fixes import unique

from resilient.logger import Logger
from resilient.selection_strategies import SelectBestK


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 60


def to_matrix(scores):
    lengths = numpy.array([len(x) for x in scores])
    ret = numpy.zeros(shape=(len(scores), lengths.max()))
    for i, score in enumerate(scores):
        ret[i, :len(score)] += score
    return ret


def format_param(param):
    if isinstance(param, int):
        return "{:5d}".format(param)
    else:
        return "{:5.3f}".format(param)


def run_experiment(dataset_name, data, target,
                   pipeline, ensemble, selection_strategy,
                   cv_method, n_iter, seed, rf, use_mcc):
    log_filename = "../results/{generator}/{selection}/{dataset}_{metric}_{date:%Y%m%d-%H%M-%S}.txt".format(
        generator=ensemble.training_strategy.train_set_generator.__class__.__name__,
        selection=selection_strategy.__class__.__name__,
        dataset=dataset_name,
        metric="mcc" if use_mcc else "acc",
        date=datetime.utcnow()
    )
    sys.stdout = Logger(log_filename)

    labels, target = unique(target, return_inverse=True)
    flt_data = pipeline.fit_transform(data) if pipeline is not None else data

    ensemble.set_params(random_state=seed)

    print HORIZ_LINE
    print "Experiment file:", log_filename
    print "Experiment seed:", seed
    print "Cross validation method:"
    print cv_method(target, n_iter, seed)
    print HORIZ_LINE
    print "Dataset name:", dataset_name
    print "Dataset size:", len(flt_data)
    print "Dataset labels:", labels
    print HORIZ_LINE
    if pipeline is not None:
        print "Pipeline:"
        print pipeline
        print HORIZ_LINE
    print "Resilient ensemble:"
    print ensemble
    print HORIZ_LINE
    print "Selection strategy:"
    print selection_strategy
    print HORIZ_LINE

    # results is a dictionary with k as key and a vector as value, containing the result for that k on each iteration
    scores_opt = numpy.zeros(n_iter)
    if selection_strategy.__class__ == SelectBestK:
        params_opt = numpy.zeros(n_iter, dtype=int)
    else:
        params_opt = numpy.zeros(n_iter)

    re_scores = []
    re_params = []
    rf_scores = numpy.zeros(n_iter)
    it = -1
    for train_indices, test_indices in cv_method(target, n_iter, seed):
        it += 1
        print "\rRunning", (it+1), "iteration..."
        train_data, train_target = flt_data[train_indices], target[train_indices]
        test_data, test_target = flt_data[test_indices], target[test_indices]
        ensemble.set_params(selection_strategy=selection_strategy)
        ensemble.fit(train_data, train_target)
        scores_opt[it] = ensemble.score(test_data, test_target, use_mcc=use_mcc)
        params_opt[it] = ensemble.selection_strategy.param

        params = ensemble.selection_strategy.get_optimization_grid(ensemble)
        if len(params) > len(re_params):
            re_params = numpy.array(params)
        re_scores.append(numpy.zeros(len(params)))
        for i, param in enumerate(params):
            print "\rTesting using param:", param,
            ensemble.selection_strategy.param = param
            re_scores[it][i] = ensemble.score(test_data, test_target, use_mcc=use_mcc)
        print ""
        if rf is not None:
            # For Random Forest, don't use the filtered data
            print "\rRunning random forest..."
            rf.fit(data[train_indices], target[train_indices])
            if use_mcc:
                rf_scores[it] = matthews_corrcoef(rf.predict(data[test_indices]), target[test_indices])
            else:
                rf_scores[it] = rf.score(data[test_indices], target[test_indices])

    re_scores = to_matrix(re_scores)
    for i, row in enumerate(transpose(re_scores)):
        print "{}: {} - Mean: {:.3f}".format(format_param(re_params[i]), row, row.mean())

    best_score_index_per_iter = re_scores.argmax(axis=1)
    best_score_per_iter = re_scores[range(n_iter), best_score_index_per_iter]
    best_param_per_iter = re_params[best_score_index_per_iter]
    print HORIZ_LINE
    print "Bst k: {} - Mean of bst k: {:.3f}".format(best_param_per_iter, best_param_per_iter.mean())
    print "Bst r: {} - Mean of bst r: {:.3f}".format(best_score_per_iter, best_score_per_iter.mean())
    print HORIZ_LINE
    print "Opt k: {} - Mean of opt k: {:.3f}".format(params_opt, params_opt.mean())
    print "Opt r: {} - Mean of opt r: {:.3f}".format(scores_opt, scores_opt.mean())
    print HORIZ_LINE
    mean_score_per_param = re_scores.mean(axis=0)
    best_mean_score_param = re_params[mean_score_per_param.argmax()]
    best_mean_score_row = re_scores[:, mean_score_per_param.argmax()]
    print "Bst m: {} - Best of means: {:.3f} (k = {})".format(best_mean_score_row, best_mean_score_row.mean(),
                                                              format_param(best_mean_score_param))
    print HORIZ_LINE
    if rf is not None:
        print "RndFr: {} - Mean: {:.3f}".format(rf_scores, rf_scores.mean())

    sys.stdout.finish()
