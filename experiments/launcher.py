from stereovis.hybrid_stereo import HybridStereoMatching


if __name__ == '__main__':
    stereo = HybridStereoMatching("/home/gdikov/StereoVision/SemiframelessStereoMatching/"
                                  "experiments/configs/pendulum.yaml")
    stereo.run()
