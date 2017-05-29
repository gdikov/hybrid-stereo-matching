from stereovis.hybrid_stereo import HybridStereoMatching


def run_experiment_from_config(config_file):
    stereo = HybridStereoMatching(config_file)
    stereo.run()
