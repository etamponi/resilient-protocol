"""
Import and define everything that can be useful during an interactive
ipython session.
"""

from glob import glob

# noinspection PyUnresolvedReferences
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from resilient.experiment import *

# noinspection PyUnresolvedReferences
from configs import ensembles, crossvals, selections
# noinspection PyUnresolvedReferences
from configs.ensembles import *
# noinspection PyUnresolvedReferences
from configs.crossvals import *
# noinspection PyUnresolvedReferences
from configs.selections import *
from resilient.result_analysis import *
from resilient.cross_validation import *

__author__ = 'tamponi'


def get_ensemble(directory):
    data = get_data(directory+"/ensemble")
    if data is not None:
        return data["ensemble"]
    else:
        return None


def get_tested_ensembles(results_dir="./results"):
    ret = []
    for d in glob(results_dir + "/*/"):
        e = get_ensemble(d)
        if e is not None:
            ret.append(e)
    return ret


def get_ensembles_without_experiments(results_dir="./results"):
    ret = []
    for d in glob(results_dir + "/*/"):
        e = get_ensemble(d)
        if e is not None:
            ensemble_dir = e.get_directory()
            experiments = glob(results_dir + "/" + ensemble_dir + "/*/*/*/")
            if len(experiments) == 0:
                ret.append(e)
    return ret


def get_ensemble_experiments(
        ensemble, dataset_prefix,
        selection_cls=None, results_dir="./results"):
    experiments = []
    ensemble_dir = ensemble.get_directory()
    for directory in glob(results_dir + "/" + ensemble_dir + "/*/*/*/"):
        experiment = get_data(directory + "/experiment")
        if experiment is None:
            continue
        dataset_name = experiment["dataset_name"]
        s_strategy_cls = type(experiment["selection_strategy"])
        if dataset_name.startswith(dataset_prefix) and \
                (selection_cls is None or selection_cls == s_strategy_cls):
            experiment["ensemble"] = ensemble
            experiments.append(experiment)
    return experiments


def get_all_experiments(dataset_prefix="", results_dir="./results"):
    experiments = {}
    for ens in get_tested_ensembles(results_dir):
        exps = get_ensemble_experiments(
            ens, dataset_prefix, selection_cls=None, results_dir=results_dir
        )
        if len(exps) > 0:
            experiments[ens] = exps
    return experiments


def join_experimental_results(experiments):
    ret = []
    for exp in experiments:
        for row in exp["results"]:
            ret.append(row)
    return numpy.array(ret)


def get_threshold_range(experiment):
    return experiment["selection_strategy"].get_threshold_range(
        experiment["ensemble"].n_estimators
    )


def plot_experiments(experiments, labels=None,
                     plot_average=False, scoring=confusion_to_accuracy):
    pyplot.hold(True)
    for i, exp in enumerate(experiments):
        x = get_threshold_range(exp)
        scores = results_to_scores(exp["results"], scoring)
        if labels is not None and len(labels) > i:
            label = labels[i]
        else:
            label = "{} {}".format(exp["dataset_name"], i+1)
        pyplot.plot(x, scores.mean(axis=0), label=label)
    if plot_average:
        x = get_threshold_range(experiments[0])
        scores = results_to_scores(
            join_experimental_results(experiments), scoring
        )
        pyplot.plot(x, scores.mean(axis=0), label="Average", linewidth=2)
    font = FontProperties()
    font.set_size("small")
    pyplot.legend(loc="lower center", ncol=4, prop=font)
    pyplot.grid()


def plot_experiments_multi_cv(list_of_list, labels=None, relative=True,
                              scoring=confusion_to_accuracy):
    pyplot.hold(True)
    for i, experiments in enumerate(list_of_list):
        x = get_threshold_range(experiments[0])
        if not relative:
            x *= experiments[0]["ensemble"].n_estimators
        scores = results_to_scores(
            join_experimental_results(experiments), scoring
        )
        if labels is not None:
            label = labels[i]
        else:
            label = str(i)
        pyplot.plot(x[::2], scores.mean(axis=0)[::2], label=label)
    pyplot.legend(loc="lower center", ncol=3)
    pyplot.grid()
