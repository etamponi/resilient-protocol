import sys
import numpy
from sklearn import cross_validation
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.utils.fixes import unique
from resilient.logger import Logger
from resilient.selection_strategies import SelectBestK

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


HORIZ_LINE = "-" * 60


def run_experiment(dataset_name, data, target, pipeline, ensemble, cv_method, n_iter, seed, log_filename, run_rf):
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
    print "Preprocessing pipeline:"
    print pipeline
    print HORIZ_LINE
    print "Resilient ensemble:"
    print ensemble
    print HORIZ_LINE

    # results is a dictionary with k as key and a vector as value, containing the result for that k on each iteration
    results = {}
    it = -1
    for train_indices, test_indices in cv_method(target, n_iter, seed):
        it += 1
        print "\rRunning", (it+1), "iteration..."
        train_data, train_target = flt_data[train_indices], target[train_indices]
        test_data, test_target = flt_data[test_indices], target[test_indices]
        ensemble.fit(train_data, train_target)
        n_estimators = len(ensemble.classifiers_)
        k_range = range(1, n_estimators+1, 2)
        if k_range[-1] < n_estimators:
            k_range.append(n_estimators)
        for k in k_range:
            ensemble.set_params(selection_strategy=SelectBestK(k))
            if k not in results:
                results[k] = numpy.zeros(n_iter)
            results[k][it] = ensemble.score(test_data, test_target)
            print "\rTesting using param:", k,
        print ""
    for k in sorted(results.keys()):
        print "{:6d}: {} - Mean: {:.3f}".format(k, results[k], results[k].mean())

    best_result_per_iter = numpy.zeros(n_iter)
    best_k_per_iter = numpy.zeros(n_iter, dtype=int)
    for it in range(n_iter):
        iter_results = numpy.array([results[k][it] for k in k_range])
        best_result_per_iter[it] = iter_results.max()
        best_k_per_iter[it] = k_range[iter_results.argmax()]
    print "Best k: {} - Mean of best k: {:d}".format(best_k_per_iter, int(best_k_per_iter.mean()))
    print "Best r: {} - Mean of best r: {:.3f}".format(best_result_per_iter, best_result_per_iter.mean())

    mean_result_per_k = numpy.array([results[k].mean() for k in k_range])
    k_of_best_mean = k_range[mean_result_per_k.argmax()]
    best_mean_result = results[k_of_best_mean]
    best_mean = best_mean_result.mean()
    print "Best m: {} -  Best of means: {:.3f} (k = {:d})".format(best_mean_result, best_mean, k_of_best_mean)
    if run_rf:
        rf = RandomForestClassifier(
            n_estimators=len(ensemble.classifiers_),
            max_features=ensemble.training_strategy.base_estimator.max_features,
            random_state=seed
        )
        print "\rRunning random forest...",
        scores = cross_validation.cross_val_score(rf, data, target, cv=cv_method(target, n_iter, seed))
        print "\r"
        print "  RF  : {} - Mean: {:.3f}".format(scores, scores.mean())

    sys.stdout.finish()
