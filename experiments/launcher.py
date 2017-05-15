from stereovis.hybrid_stereo import HybridStereoMatching


if __name__ == '__main__':
    stereo = HybridStereoMatching("configs/pendulum.yaml")
    stereo.run()
