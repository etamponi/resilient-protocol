from datetime import datetime
from itertools import product
import sys

import numpy
from numpy.core.fromnumeric import transpose
from sklearn import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.fixes import unique

from resilient.logger import Logger


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 60


def to_matrix(scores):
    lengths = numpy.array([len(x) for x in scores])
    ret = numpy.zeros(shape=(len(scores), lengths.max()))
    for i, score in enumerate(scores):
        ret[i, :len(score)] += score
    return ret


def prepare_params_list(selection_strategy):
    ranges = selection_strategy.get_params_ranges()
    keys = selection_strategy.get_params_names()
    ranges = [ranges[key] for key in keys]
    tuples = list(product(*ranges))
    params = [{key: t[i] for i, key in enumerate(keys)} for t in tuples]
    return keys, params


def run_experiment(dataset_name, data, target,
                   pipeline, ensemble, selection_strategy,
                   cv_method, n_iter, seed, rf, use_mcc):
    sys.stdout = Logger()

    labels, target = unique(target, return_inverse=True)
    flt_data = pipeline.fit_transform(data) if pipeline is not None else data

    ensemble.set_params(random_state=seed)

    print HORIZ_LINE
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

    scores_opt = numpy.zeros(n_iter)
    params_opt = [None] * n_iter
    re_scores = []
    keys, re_params = prepare_params_list(selection_strategy)

    rf_scores = numpy.zeros(n_iter)

    for it, (train_indices, test_indices) in enumerate(cv_method(target, n_iter, seed)):
        print "\rRunning", (it+1), "iteration..."
        train_data, train_target = flt_data[train_indices], target[train_indices]
        test_data, test_target = flt_data[test_indices], target[test_indices]
        ensemble.set_params(selection_strategy=clone(selection_strategy))
        ensemble.fit(train_data, train_target)
        scores_opt[it] = ensemble.score(test_data, test_target, use_mcc=use_mcc)
        params_opt[it] = ensemble.selection_strategy.params_to_string()

        re_scores.append(numpy.zeros(len(re_params)))
        for i, params in enumerate(re_params):
            print "\rTesting using params: {}".format(selection_strategy.params_to_string(params, join=" ")),
            ensemble.selection_strategy.params = params
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
        print "{}: {} - Mean: {:.3f}".format(selection_strategy.params_to_string(re_params[i], join=" "),
                                             row, row.mean())

    best_score_index_per_iter = re_scores.argmax(axis=1)
    best_score_per_iter = re_scores[range(n_iter), best_score_index_per_iter]
    best_param_per_iter = numpy.transpose(numpy.array([
        selection_strategy.params_to_string(re_params[i]) for i in best_score_index_per_iter
    ]))
    print HORIZ_LINE
    padding = " " * (len(selection_strategy.params_to_string(re_params[0], join=" ")) - 6)
    for i, params_row in enumerate(best_param_per_iter):
        print "{}Bt p{}: |{}| - {}".format(padding, i+1, "|".join(params_row), keys[i])
    print "{}Bst r: {} - Mean of bst r: {:.3f}".format(padding, best_score_per_iter, best_score_per_iter.mean())
    print HORIZ_LINE
    for i, params_row in enumerate(numpy.transpose(params_opt)):
        print "{}Op p{}: |{}| - {}".format(padding, i+1, "|".join(params_row), keys[i])
    print "{}Opt r: {} - Mean of opt r: {:.3f}".format(padding, scores_opt, scores_opt.mean())
    print HORIZ_LINE
    mean_score_per_param = re_scores.mean(axis=0)
    best_mean_score_param = re_params[mean_score_per_param.argmax()]
    best_mean_score_row = re_scores[:, mean_score_per_param.argmax()]
    print "{}Bst m: {} - Best of means: {:.3f} (params = {})".format(
        padding,
        best_mean_score_row, best_mean_score_row.mean(),
        selection_strategy.params_to_string(best_mean_score_param, join=" ")
    )
    print HORIZ_LINE
    if rf is not None:
        print "{}RndFr: {} - Mean: {:.3f}".format(padding, rf_scores, rf_scores.mean())

    log_filename = "../results/{generator}/{selection}/{dataset}_{metric}_{date:%Y%m%d-%H%M-%S}".format(
        generator=ensemble.training_strategy.train_set_generator.__class__.__name__,
        selection=selection_strategy.__class__.__name__,
        dataset=dataset_name,
        metric="mcc" if use_mcc else "acc",
        date=datetime.utcnow()
    )
    sys.stdout.finish(log_filename, scores=re_scores, params=re_params, params_opt=params_opt)
