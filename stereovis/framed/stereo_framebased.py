import numpy as np
from stereovis.framed.algorithms.mrf import StereoMRF


class FramebasedStereoMatching:
    def __init__(self, resolution, max_disparity, algorithm='mrf', frames=None):
        if algorithm == 'mrf':
            self.algorithm = StereoMRF(dim=resolution, n_levels=max_disparity)
        else:
            raise NotImplementedError("Only MRF is supported.")
        if frames is not None:
            # this means that the operational mode is offline and hence one can initialise the frame iterator
            self.frames_left = np.asarray(frames['left'])
            self.frames_right = np.asarray(frames['right'])
            self.frames_timestamps = np.asarray(frames['ts'])

    def get_output(self):
        return self.algorithm.get_map_belief()

    def run_next_frame(self, image_left, image_right, prior=None):
        """
        Run one single frame of the frame-based stereo matching. Should be used when running online.
        
        Args:
            image_left: a numpy array representing the left image 
            image_right: a numpy array representing the right image
            prior: optional, a numpy array with disparity values 

        Returns:
            In-place method.
        """
        self.algorithm.loop_belief(image_left=image_left,
                                   image_right=image_right,
                                   prior=prior,
                                   n_iter=5,
                                   reinit_messages=False)

    def run(self, prior_info=None):
        """
        Run the frame-based stereo matching on all frames and priors.
        
        Args:
            prior_info: optional, a list of priors a subset of which is used to initialise the algorithm.  

        Returns:
            
        """
        prior_indices = [np.searchsorted(prior_info['ts'], t_frame, side="right") for t_frame in self.frames_timestamps]
        priors = prior_info['priors'][prior_indices]
        assert len(priors) == len(self.frames_left) == len(self.frames_right)
        for left, right, prior in zip(self.frames_left, self.frames_right, priors):
            self.run_next_frame(left, right, prior)
