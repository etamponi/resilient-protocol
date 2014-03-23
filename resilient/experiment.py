import hashlib
from multiprocessing import Pool
import os
import signal
import cPickle

import numpy
from numpy.core.fromnumeric import transpose
from sklearn import clone
from sklearn.externals.joblib.parallel import multiprocessing
from sklearn.metrics.metrics import confusion_matrix
from sklearn.utils.fixes import unique
from sklearn.utils.random import check_random_state

from resilient.logger import Logger
from resilient.result_analysis import results_to_scores, confusion_to_accuracy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 88


def get_data_filename(ensemble, seed, cv_method, n_iter, dataset_name, results_dir):
    cv_class = cv_method([0, 1]*(100*n_iter), n_iter, seed).__class__.__name__
    ensemble_dir = hashlib.md5(repr(ensemble)).hexdigest()
    filename = "{results_dir}/{ensemble_dir}/{dataset_name}/{cv_class}_{n_iter:02d}".format(
        results_dir=results_dir,
        ensemble_dir=ensemble_dir,
        dataset_name=dataset_name,
        cv_class=cv_class,
        n_iter=n_iter
    )
    return filename


def check_already_present(filename):
    filename += ".dat"
    if os.path.isfile(filename):
        with open(filename) as f:
            return cPickle.load(f)
    else:
        return None


def run_cv_iter((ensemble, inp, y, train_indices, test_indices, threshold_range, it)):
    Logger.get().write("!Running", (it+1), "iteration...")

    train_inp, train_y = inp[train_indices], y[train_indices]
    test_inp, test_y = inp[test_indices], y[test_indices]

    ensemble.fit(train_inp, train_y)

    confusion_matrices = numpy.zeros((len(threshold_range), ensemble.n_classes_, ensemble.n_classes_))
    for i, threshold in enumerate(threshold_range):
        ensemble.selection_strategy.threshold = threshold
        Logger.get().write("!Testing using threshold: {:.3f}".format(threshold))
        confusion_matrices[i] = confusion_matrix(ensemble.predict(test_inp), test_y)
    return confusion_matrices


def run_cv(ensemble, seed, cv_method, n_iter, dataset_name, inp, y, threshold_range, results_dir, run_async):
    ensemble.set_params(random_state=seed)
    labels, y = unique(y, return_inverse=True)
    y = labels[y]

    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Random state seed:", seed)
    Logger.get().write("Cross validation method:", cv_method(y, n_iter, seed).__class__.__name__)
    Logger.get().write("Cross validation iterations:", n_iter)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Dataset name:", dataset_name)
    Logger.get().write("Dataset size:", len(inp))
    Logger.get().write("Dataset labels:", labels)
    Logger.get().write(HORIZ_LINE)
    Logger.get().write("Resilient ensemble:")
    Logger.get().write(ensemble)
    Logger.get().write(HORIZ_LINE)

    filename = get_data_filename(ensemble, seed, cv_method, n_iter, dataset_name, results_dir)
    data = check_already_present(filename)
    if data is not None:
        # noinspection PyTypeChecker
        # Check if the threshold_range in the saved data is the same
        if any(data["threshold_range"] != threshold_range):
            # If it is different, we need to recompute...
            data = None

    if data is None:
        # Do an initial shuffle of the data, as the cv_method could be deterministic
        indices = check_random_state(seed).permutation(len(inp))
        inp, y = inp[indices], y[indices]

        args = [(clone(ensemble), numpy.copy(inp), numpy.copy(y), lix, tix, numpy.copy(threshold_range), it)
                for it, (lix, tix) in enumerate(cv_method(y, n_iter, seed))]

        if run_async:
            def init_worker():
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            os.system('taskset -p 0xffffffff %d &> /dev/null' % os.getpid())
            pool = Pool(min(multiprocessing.cpu_count()-1, len(args)/2), initializer=init_worker)
            try:
                results = numpy.array(pool.map_async(run_cv_iter, args).get(1000000))
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
        else:
            results = numpy.array(map(run_cv_iter, args))
    else:
        results = data["results"]

    scores = results_to_scores(results, confusion_to_accuracy)
    for i, row in enumerate(transpose(scores)):
        Logger.get().write("{:11.3f}: {} - Mean score: {:.3f}".format(threshold_range[i], row, row.mean()))

    best_score_index_per_iter = scores.argmax(axis=1)
    best_score_per_iter = scores[range(n_iter), best_score_index_per_iter]
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
    Logger.get().write("Best mean s: {} - Best of mean scores: {:.3f} (threshold = {:.3f})".format(
        best_mean_score_row, best_mean_score_row.mean(), best_mean_score_param
    ))
    Logger.get().write(HORIZ_LINE)

    Logger.get().save(filename, ensemble=ensemble, results=results, threshold_range=threshold_range)
    return results
