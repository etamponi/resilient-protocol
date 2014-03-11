from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
import os

import numpy
from numpy.core.fromnumeric import transpose
import signal
from sklearn import clone
from sklearn.externals.joblib.parallel import multiprocessing
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.fixes import unique

from resilient.logger import Logger


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 60


def run_iter((ensemble, re_params, it, train_indices, test_indices, data, flt_data, target, rf, use_mcc)):
    Logger.get().write("!Running", (it+1), "iteration...")
    train_data, train_target = flt_data[train_indices], target[train_indices]
    test_data, test_target = flt_data[test_indices], target[test_indices]
    ensemble.fit(train_data, train_target)

    score_opt = ensemble.score(test_data, test_target, use_mcc=use_mcc)
    param_opt = ensemble.selection_strategy.params_to_string()

    re_scores = numpy.zeros(len(re_params))
    for i, params in enumerate(re_params):
        ensemble.selection_strategy.params = params
        Logger.get().write("!Testing using params: {}".format(ensemble.selection_strategy.params_to_string(join=" ")))
        re_scores[i] = ensemble.score(test_data, test_target, use_mcc=use_mcc)

    if rf is not None:
        # For Random Forest, don't use the filtered data
        Logger.get().write("!Running random forest...")
        rf.fit(data[train_indices], target[train_indices])
        if use_mcc:
            rf_score = matthews_corrcoef(rf.predict(data[test_indices]), target[test_indices])
        else:
            rf_score = rf.score(data[test_indices], target[test_indices])
    else:
        rf_score = None

    return {
        "score_opt": score_opt,
        "param_opt": param_opt,
        "re_scores": re_scores,
        "rf_score": rf_score
    }


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_experiment(dataset_name, data, target, pipeline, ensemble, cv_method, n_iter, seed, rf, use_mcc, results_dir):
    labels, target = unique(target, return_inverse=True)
    flt_data = pipeline.fit_transform(data) if pipeline is not None else data
    selection_strategy = ensemble.selection_strategy

    ensemble.set_params(random_state=seed)

    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Experiment seed:", seed)
    Logger.get().write("Cross validation method:")
    Logger.get().write(cv_method(target, n_iter, seed))
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Dataset name:", dataset_name)
    Logger.get().write("Dataset size:", len(flt_data))
    Logger.get().write("Dataset labels:", labels)
    Logger.get().write(HORIZ_LINE)
    if pipeline is not None:
        Logger.get().write("Pipeline:")
        Logger.get().write(pipeline)
        Logger.get().write(HORIZ_LINE)
    Logger.get().write("Resilient ensemble:")
    Logger.get().write(ensemble)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Selection strategy:")
    Logger.get().write(selection_strategy)
    Logger.get().write(HORIZ_LINE)

    keys, re_params = ensemble.selection_optimizer.build_params_matrix(selection_strategy, matrix_form=False)
    re_params = [{keys[i]: params[i]} for i in range(len(keys)) for params in re_params]
    re_scores = numpy.zeros((n_iter, len(re_params)))
    scores_opt = numpy.zeros(n_iter)
    params_opt = [None] * n_iter
    rf_scores = numpy.zeros(n_iter)

    args = list((clone(ensemble), deepcopy(re_params),
                 it, lix, tix, numpy.copy(data), numpy.copy(flt_data), numpy.copy(target),
                 None if rf is None else clone(rf), use_mcc)
                for it, (lix, tix) in enumerate(cv_method(target, n_iter, seed)))

    os.system('taskset -p 0xffffffff %d' % os.getpid())
    pool = Pool(min(multiprocessing.cpu_count()-1, len(args)), initializer=init_worker)
    try:
        results = pool.map_async(run_iter, args).get(1000000)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise

    for it, result in enumerate(results):
        re_scores[it] = result["re_scores"]
        scores_opt[it] = result["score_opt"]
        params_opt[it] = result["param_opt"]
        if rf is not None:
            rf_scores[it] = result["rf_score"]

    for i, row in enumerate(transpose(re_scores)):
        Logger.get().write("{}: {} - Mean: {:.3f}".format(selection_strategy.params_to_string(re_params[i], join=" "),
                                                          row, row.mean()))

    best_score_index_per_iter = re_scores.argmax(axis=1)
    best_score_per_iter = re_scores[range(n_iter), best_score_index_per_iter]
    best_param_per_iter = numpy.transpose(numpy.array([
        selection_strategy.params_to_string(re_params[i]) for i in best_score_index_per_iter
    ]))
    Logger.get().write(HORIZ_LINE)
    padding = " " * (len(selection_strategy.params_to_string(re_params[0], join=" ")) - 6)
    for i, params_row in enumerate(best_param_per_iter):
        Logger.get().write("{}Bt p{}: |{}| - {}".format(padding, i+1, "|".join(params_row), keys[i]))
    Logger.get().write("{}Bst r: {} - Mean of bst r: {:.3f}".format(
        padding, best_score_per_iter, best_score_per_iter.mean()
    ))
    Logger.get().write(HORIZ_LINE)
    for i, params_row in enumerate(numpy.transpose(params_opt)):
        Logger.get().write("{}Op p{}: |{}| - {}".format(padding, i+1, "|".join(params_row), keys[i]))
    Logger.get().write("{}Opt r: {} - Mean of opt r: {:.3f}".format(padding, scores_opt, scores_opt.mean()))
    Logger.get().write(HORIZ_LINE)
    mean_score_per_param = re_scores.mean(axis=0)
    best_mean_score_param = re_params[mean_score_per_param.argmax()]
    best_mean_score_row = re_scores[:, mean_score_per_param.argmax()]
    Logger.get().write("{}Bst m: {} - Best of means: {:.3f} (params = {})".format(
        padding,
        best_mean_score_row, best_mean_score_row.mean(),
        selection_strategy.params_to_string(best_mean_score_param, join=" ")
    ))
    Logger.get().write(HORIZ_LINE)
    if rf is not None:
        Logger.get().write("{}RndFr: {} - Mean: {:.3f}".format(padding, rf_scores, rf_scores.mean()))

    log_filename = "../{results_dir}/{generator}/{selection}/{dataset}_{metric}_{date:%Y%m%d-%H%M-%S}".format(
        generator=ensemble.training_strategy.train_set_generator.__class__.__name__,
        selection=selection_strategy.__class__.__name__,
        dataset=dataset_name,
        metric="mcc" if use_mcc else "acc",
        date=datetime.utcnow(),
        results_dir=results_dir
    )

    Logger.get().save(log_filename, scores=re_scores, params=re_params, params_opt=params_opt)
