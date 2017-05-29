from experiments.launcher import run_experiment_from_config


if __name__ == '__main__':
    import pip
    pip.main(['install', 'pyyaml'])
    run_experiment_from_config("experiments/configs/hybrid/head.yaml")
