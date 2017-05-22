import numpy as np
from stereovis.framed.algorithms.mrf import StereoMRF

import time
from spinn_machine.utilities.progress_bar import ProgressBar
import logging

logger = logging.getLogger(__file__)


class FramebasedStereoMatching:
    def __init__(self, resolution, max_disparity, algorithm='mrf', frames=None):
        if algorithm == 'mrf':
            # reverse the resolution order since x-dimension corresponds to n_cols and y to n_rows
            # and the shape initialisation of numpy is (n_rows, n_cols) which is (y, x)
            x, y = resolution
            self.algorithm = StereoMRF(dim=(y, x), n_levels=max_disparity)
        else:
            raise NotImplementedError("Only MRF is supported.")
        if frames is not None:
            # this means that the operational mode is offline and hence one can initialise the frame iterator
            self.frames_left = np.asarray(frames['left'])
            self.frames_right = np.asarray(frames['right'])
            self.frames_timestamps = np.asarray(frames['ts'])
            # initialise the placeholder for the depth-resolved frames
            self.depth_frames = []

    def get_timestamps(self):
        return self.frames_timestamps

    def get_output(self):
        self.depth_frames = np.asarray(self.depth_frames)
        return self.depth_frames

    def run_next_frame(self, image_left, image_right, prior=None):
        """
        Run one single frame of the frame-based stereo matching. Should be used when running online.
        
        Args:
            image_left: a numpy array representing the left image 
            image_right: a numpy array representing the right image
            prior: optional, a numpy array with disparity values 

        Returns:
            A numpy array representing the depth map resolved by the algorithm.
        """
        self.algorithm.loop_belief(image_left=image_left,
                                   image_right=image_right,
                                   prior=prior,
                                   n_iter=10,
                                   reinit_messages=True)
        depth_map = self.algorithm.get_map_belief()
        self.depth_frames.append(depth_map)
        return depth_map

    def run(self, prior_info=None):
        """
        Run the frame-based stereo matching on all frames and priors.
        
        Args:
            prior_info: optional, a list of priors a subset of which is used to initialise the algorithm.  

        Returns:
            
        """
        n_frames = len(self.frames_timestamps)
        if prior_info is not None:
            if len(prior_info['ts']) > n_frames:
                # pick the n closest ones (where n is the number of frames)
                prior_indices = [np.searchsorted(prior_info['ts'], t_frame, side="left") for t_frame in self.frames_timestamps]
                priors = prior_info['priors'][prior_indices]
            else:
                priors = prior_info['priors']
            assert len(priors) == len(self.frames_left) == len(self.frames_right)

            pb = ProgressBar(n_frames, "Starting offline frame-based stereo matching with prior initialisation.")
            start_timer = time.time()
            for left, right, prior in zip(self.frames_left, self.frames_right, priors):
                self.run_next_frame(left, right, prior)
                pb.update()
            end_timer = time.time()
            pb.end()
        else:
            pb = ProgressBar(n_frames, "Starting offline frame-based stereo matching without prior initialisation.")
            start_timer = time.time()
            for left, right in zip(self.frames_left, self.frames_right):
                self.run_next_frame(left, right)
                pb.update()
            end_timer = time.time()
            pb.end()
        logger.info("Frame-based stereo matching took {}s per image pair on average.".format((end_timer - start_timer)
                                                                                             / n_frames))
