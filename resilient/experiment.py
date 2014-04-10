from multiprocessing import Pool
import os
import signal
import cPickle

import arff
import numpy
from sklearn import clone
from sklearn.externals.joblib.parallel import multiprocessing
from sklearn.metrics.metrics import confusion_matrix

from resilient.logger import Logger
from resilient.result_analysis import results_to_scores, confusion_to_accuracy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 88


def get_ensemble_filename(ensemble, results_dir="./results"):
    return "{}/{}".format(results_dir, ensemble.get_filename())


def get_ensemble_dir(ensemble, results_dir="./results"):
    return "{}/{}".format(results_dir, ensemble.get_directory())


def get_experiment_filename(ensemble, selection_strategy,
                            dataset_name, cross_validation,
                            results_dir="./results"):
    filename = "{ensemble_dir}/{selection}/{dataset}/{cv}/experiment".format(
        ensemble_dir=get_ensemble_dir(ensemble, results_dir),
        selection=selection_strategy.__class__.__name__,
        dataset=dataset_name,
        cv=cross_validation.get_filename()
    )
    return filename


def get_data(filename):
    filename += ".dat"
    if os.path.isfile(filename):
        with open(filename) as f:
            return cPickle.load(f)
    else:
        return None


def _run_cv_iter((ensemble, selection_strategy, inp, y,
                  train_indices, test_indices, seed, it)):
    Logger.get().write("!Running", (it+1), "iteration...")

    train_inp, train_y = inp[train_indices], y[train_indices]
    test_inp, test_y = inp[test_indices], y[test_indices]

    ensemble.set_params(
        random_state=seed, selection_strategy=selection_strategy
    )
    ensemble.fit(train_inp, train_y)

    threshold_range = selection_strategy.get_threshold_range(
        ensemble.n_estimators
    )
    confusion_matrices = numpy.zeros(
        (len(threshold_range), ensemble.n_classes_, ensemble.n_classes_)
    )
    for i, threshold in enumerate(threshold_range):
        ensemble.selection_strategy.threshold = threshold
        Logger.get().write("!Testing using threshold: {:.3f}".format(threshold))
        confusion_matrices[i] = confusion_matrix(
            ensemble.predict(test_inp), test_y
        )
    return confusion_matrices


def load_dataset(dataset_name, datasets_dir="./datasets"):
    with open("{}/{}.arff".format(datasets_dir, dataset_name)) as f:
        d = arff.load(f)
        inp = numpy.array([row[:-1] for row in d['data']])
        y = numpy.array([row[-1] for row in d['data']])
    labels, y = numpy.unique(y, return_inverse=True)
    y = labels[y]
    # "---" is a special label used by ReadyKFold, check there
    labels = labels[labels != "---"]
    return inp, y, labels


def run_experiment(ensemble, selection_strategy, dataset_name, cross_validation,
                   datasets_dir="./datasets", results_dir="./results",
                   run_async=False):
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Dataset name:", dataset_name)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Cross validation:")
    Logger.get().write(cross_validation)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Resilient ensemble:")
    Logger.get().write(ensemble)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Selection strategy:")
    Logger.get().write(selection_strategy)
    Logger.get().write(HORIZ_LINE)

    inp, y, labels = load_dataset(dataset_name, datasets_dir)

    filename = get_experiment_filename(
        ensemble, selection_strategy, dataset_name,
        cross_validation, results_dir
    )
    data = get_data(filename)

    if data is None:
        Logger.get().save(
            get_ensemble_filename(ensemble, results_dir), ensemble=ensemble
        )

        if run_async:
            def init_worker():
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            args = [
                (clone(ensemble), clone(selection_strategy), inp, y,
                 train_indices, test_indices, seed, it)
                for it, (seed, train_indices, test_indices)
                in enumerate(cross_validation.build(y))
            ]
            os.system('taskset -p 0xffffffff %d &> /dev/null' % os.getpid())
            processes = min(multiprocessing.cpu_count()-1, len(args)/2)
            pool = Pool(processes, initializer=init_worker)
            try:
                results = numpy.array(
                    pool.map_async(_run_cv_iter, args).get(1000000)
                )
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
        else:
            ensemble = clone(ensemble)
            args = [
                (ensemble, selection_strategy, inp, y,
                 train_indices, test_indices, seed, it)
                for it, (seed, train_indices, test_indices)
                in enumerate(cross_validation.build(y))
            ]
            results = numpy.array(map(_run_cv_iter, args))
    else:
        results = data["results"]

    Logger.get().write(HORIZ_LINE)
    threshold_range = selection_strategy.get_threshold_range(
        ensemble.n_estimators
    )
    scores = results_to_scores(results, confusion_to_accuracy)
    for i, row in enumerate(numpy.transpose(scores)):
        Logger.get().write(
            "{:11.3f}: {} - Mean score: {:.3f}".format(
                threshold_range[i], row, row.mean()
            )
        )

    best_score_index_per_iter = scores.argmax(axis=1)
    best_score_per_iter = scores[
        range(cross_validation.total_runs()), best_score_index_per_iter
    ]
    best_param_per_iter = threshold_range[best_score_index_per_iter]
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Best thresh: {}".format(best_param_per_iter))
    Logger.get().write("Best scores: {} - Mean of best scores: {:.3f}".format(
        best_score_per_iter, best_score_per_iter.mean()
    ))
    Logger.get().write(HORIZ_LINE)
    mean_score_per_param = scores.mean(axis=0)
    best_mean_score_param = threshold_range[mean_score_per_param.argmax()]
    best_mean_score_row = scores[:, mean_score_per_param.argmax()]
    Logger.get().write(
        "Best mean s: {} - Best of mean scores: {:.3f} (threshold = {:.3f})"
        .format(
            best_mean_score_row, best_mean_score_row.mean(),
            best_mean_score_param
        )
    )
    Logger.get().write(HORIZ_LINE)

    Logger.get().save(filename, results=results,
                      selection_strategy=selection_strategy,
                      dataset_name=dataset_name,
                      cross_validation=cross_validation)

    Logger.get().clear()
    return data
