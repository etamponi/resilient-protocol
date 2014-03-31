#!/usr/bin/env python

from IPython.external import argparse

from resilient.experiment import run_experiment


__author__ = 'tamponi'


def get_config(config_module, variable):
    module = __import__(config_module, fromlist=variable)
    return module.__dict__[variable]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generator_params")
    parser.add_argument("datasets")
    parser.add_argument("-ss", "--selection-strategy", default="best")
    parser.add_argument("-cv", "--cross-validation", default="not_nested_10fold")
    parser.add_argument("-eg", "--ensemble-generator", default="generalized_exponential_ensemble")
    parser.add_argument("-rd", "--results-dir", default="./results")
    parser.add_argument("-dd", "--datasets-dir", default="./datasets")
    parser.add_argument("-em", "--ensemble-module", default="configs.ensembles")
    parser.add_argument("-sm", "--selection-module", default="configs.selections")
    parser.add_argument("-cm", "--crossval-module", default="configs.crossvals")
    parser.add_argument("-a", "--async", action="store_true")
    args = parser.parse_args()

    selection_strategy = get_config(args.selection_module, args.selection_strategy)
    cross_validation = get_config(args.crossval_module, args.cross_validation)

    exec "from {} import {}".format(args.ensemble_module, args.ensemble_generator)
    ensemble = eval("{}({})".format(args.ensemble_generator, args.generator_params))

    for dataset in args.datasets.split(" "):
        run_experiment(
            ensemble=ensemble,
            selection_strategy=selection_strategy,
            dataset_name=dataset,
            cross_validation=cross_validation,
            datasets_dir=args.datasets_dir,
            results_dir=args.results_dir,
            run_async=args.async
        )
