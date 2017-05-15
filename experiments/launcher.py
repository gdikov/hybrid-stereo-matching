from stereovis.hybrid_stereo import HybridStereoMatching


if __name__ == '__main__':
    stereo = HybridStereoMatching("configs/nst_letters.yaml")
    stereo.run()
