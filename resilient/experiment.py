from datetime import datetime
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


def run_experiment(dataset_name, data, target, pipeline, ensemble, cv_method, n_iter, seed, rf, use_mcc):
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

    scores_opt = numpy.zeros(n_iter)
    params_opt = [None] * n_iter
    re_scores = []
    keys, re_params = ensemble.selection_optimizer.build_params_matrix(selection_strategy, matrix_form=False)

    rf_scores = numpy.zeros(n_iter)

    for it, (train_indices, test_indices) in enumerate(cv_method(target, n_iter, seed)):
        Logger.get().write("!Running", (it+1), "iteration...")
        train_data, train_target = flt_data[train_indices], target[train_indices]
        test_data, test_target = flt_data[test_indices], target[test_indices]
        ensemble.set_params(selection_strategy=clone(selection_strategy))
        ensemble.fit(train_data, train_target)
        scores_opt[it] = ensemble.score(test_data, test_target, use_mcc=use_mcc)
        params_opt[it] = ensemble.selection_strategy.params_to_string()

        re_scores.append(numpy.zeros(len(re_params)))
        for i, params in enumerate(re_params):
            Logger.get().write("!Testing using params: {}".format(
                selection_strategy.params_to_string(params, join=" ")
            ))
            ensemble.selection_strategy.params = params
            re_scores[it][i] = ensemble.score(test_data, test_target, use_mcc=use_mcc)

        if rf is not None:
            # For Random Forest, don't use the filtered data
            Logger.get().write("!Running random forest...")
            rf.fit(data[train_indices], target[train_indices])
            if use_mcc:
                rf_scores[it] = matthews_corrcoef(rf.predict(data[test_indices]), target[test_indices])
            else:
                rf_scores[it] = rf.score(data[test_indices], target[test_indices])

    re_scores = to_matrix(re_scores)
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

    log_filename = "../results/{generator}/{selection}/{dataset}_{metric}_{date:%Y%m%d-%H%M-%S}".format(
        generator=ensemble.training_strategy.train_set_generator.__class__.__name__,
        selection=selection_strategy.__class__.__name__,
        dataset=dataset_name,
        metric="mcc" if use_mcc else "acc",
        date=datetime.utcnow()
    )

    sys.stdout.save(log_filename, scores=re_scores, params=re_params, params_opt=params_opt)
