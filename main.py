from resilient import experiment

__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


if __name__ == "__main__":
    from experiments import humvar_grid_config as cfg
    experiment.run_experiment(**cfg.config)
