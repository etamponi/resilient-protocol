# Import and define everything that can be useful during an interactive ipython session

from glob import glob

# noinspection PyUnresolvedReferences
import numpy
from resilient.experiment import get_data

# noinspection PyUnresolvedReferences
from configs.ensembles import *
# noinspection PyUnresolvedReferences
from configs.crossvals import *
# noinspection PyUnresolvedReferences
from configs.selections import *
from resilient.result_analysis import *

__author__ = 'tamponi'


def get_ensemble(directory):
    data = get_data(directory+"/ensemble")
    if data is not None:
        return data["ensemble"]
    else:
        return None


def get_tested_ensembles(results_dir="./results"):
    ensembles = []
    for d in glob(results_dir + "/*/"):
        e = get_ensemble(d)
        if e is not None:
            ensembles.append(e)
    return ensembles


def get_ensemble_experiments(ensemble, results_dir="./results"):
    experiments = []
    ensemble_dir = ensemble.get_directory()
    for directory in glob(results_dir + "/" + ensemble_dir + "/*/*/*/"):
        experiment = get_data(directory + "/experiment")
        if experiment is not None:
            experiment["ensemble"] = ensemble
        experiments.append(experiment)
    return experiments


def get_all_experiments(results_dir="./results"):
    ensembles = get_tested_ensembles(results_dir)
    experiments = []
    for ens in ensembles:
        experiments += get_ensemble_experiments(ens, results_dir)
    return experiments, ensembles


def join_results(*all_results):
    ret = []
    for results in all_results:
        for row in results:
            ret.append(row)
    return numpy.array(ret)


def get_threshold_range(experiment):
    return experiment["selection_strategy"].get_threshold_range(experiment["ensemble"].n_estimators)
