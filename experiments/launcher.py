from stereovis.hybrid_stereo import HybridStereoMatching


def run_pendulum():
    stereo = HybridStereoMatching("configs/pendulum.yaml")
    stereo.run()


if __name__ == '__main__':
    run_pendulum()
