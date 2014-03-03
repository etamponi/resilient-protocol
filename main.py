__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


if __name__ == "__main__":
    from resilient import experiment_threshold as exp
    from experiments import humvar_gridpdf_config as cfg
    exp.run_experiment(**cfg.config)
