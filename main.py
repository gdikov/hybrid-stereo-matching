import logging
logging.basicConfig(level=logging.INFO)

from experiments.launcher import run_experiment_from_config

if __name__ == '__main__':
    # run_experiment_from_config("experiments/configs/hybrid/head_downsampled.yaml")
    run_experiment_from_config("experiments/configs/hybrid/checkerboard_downsampled.yaml")
    # run_experiment_from_config("experiments/configs/hybrid/boxes_and_cones_downsampled.yaml")
