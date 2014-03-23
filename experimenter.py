import sys

import arff
from sympy.utilities.tests.test_lambdify import numpy

from resilient.experiment import run_cv


__author__ = 'tamponi'


def get_config(cfg_module_name):
    module = __import__(cfg_module_name, fromlist="config")
    return module.config


if __name__ == "__main__":
    config_module_name, dataset_name = sys.argv[1:]

    with open("./datasets/{}.arff".format(dataset_name)) as f:
        d = arff.load(f)
        inp = numpy.array([row[:-1] for row in d['data']])
        y = numpy.array([row[-1] for row in d['data']])

    run_cv(dataset_name=dataset_name, inp=inp, y=y, results_dir="./results", **get_config(config_module_name))
