#!/usr/bin/env python
"""
Command line interface for running experiments on Resilient Ensembles.

The purpose of this script is to provide a sane interface to the
:func:`resilient.experiment.run_experiment` function.

To run the script, you must specify a piece of python code as first
argument, which represents the parameters passed to an *ensemble generator*,
which is any kind of python function that returns a
:class:`resilient.ensemble.ResilientEnsemble` object.

The second parameter is a space separated list of dataset names on which you
want to test the resilient ensemble.

You can also optionally specify the ensemble generator function to call, the
module in which you defined the generator, the selection strategy to test and
the cross validation strategy to use.

The -a option activates the multiprocessing module and runs your experiments
asynchronously.
"""

from IPython.external import argparse

from resilient.experiment import run_experiment


__author__ = 'tamponi'


def _get_config(config_module, variable):
    module = __import__(config_module, fromlist=variable)
    return module.__dict__[variable]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generator_params")
    parser.add_argument("datasets")
    parser.add_argument("-ss", "--selection-strategy", default="select_best")
    parser.add_argument("-cv", "--cross-validation",
                        default="not_nested_10fold")
    parser.add_argument("-eg", "--ensemble-generator",
                        default="generalized_exponential_resilient_forest")
    parser.add_argument("-rd", "--results-dir", default="./results")
    parser.add_argument("-dd", "--datasets-dir", default="./datasets")
    parser.add_argument("-gm", "--generator-module",
                        default="configs.ensembles")
    parser.add_argument("-sm", "--selection-module",
                        default="configs.selections")
    parser.add_argument("-cm", "--crossval-module", default="configs.crossvals")
    parser.add_argument("-a", "--async", action="store_true")
    args = parser.parse_args()

    selection_strategy = _get_config(args.selection_module,
                                     args.selection_strategy)
    cross_validation = _get_config(args.crossval_module, args.cross_validation)

    exec "from {} import {}".format(args.generator_module,
                                    args.ensemble_generator)
    ensemble = eval("{}({})".format(args.ensemble_generator,
                                    args.generator_params))

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


if __name__ == "__main__":
    main()
