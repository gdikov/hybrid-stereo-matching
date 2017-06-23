import numpy as np
import logging
import time

from stereovis.framed.algorithms import StereoMRF
from spinn_machine.utilities.progress_bar import ProgressBar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import restoration, filters, morphology


logger = logging.getLogger(__file__)


class FramebasedStereoMatching(object):
    def __init__(self, resolution, max_disparity, algorithm='mrf', inputs=None):
        if algorithm == 'mrf':
            # reverse the resolution order since x-dimension corresponds to n_cols and y to n_rows
            # and the shape initialisation of numpy is (n_rows, n_cols) which is (y, x)
            x, y = resolution
            self.algorithm = StereoMRF(dim=(y, x), n_levels=max_disparity)
            if inputs is not None:
                # this means that the operational mode is offline and hence one can initialise the frame iterator
                self.frames_left = np.asarray(inputs['left'])
                self.frames_right = np.asarray(inputs['right'])
                self.frames_timestamps = np.asarray(inputs['ts'])
                # initialise the placeholder for the depth-resolved inputs
                self.depth_frames = []
        else:
            raise NotImplementedError("Only MRF is supported.")

    def get_timestamps(self):
        return self.frames_timestamps

    def get_output(self):
        self.depth_frames = np.asarray(self.depth_frames)
        return self.depth_frames

    def run_one_frame(self, image_left, image_right, prior=None):
        """
        Run one single frame of the frame-based stereo matching. Should be used when running online.
        
        Args:
            image_left: a numpy array representing the left image 
            image_right: a numpy array representing the right image
            prior: optional, a numpy array with disparity values 

        Returns:
            A numpy array representing the depth map resolved by the algorithm.
        """
        depth_map = self.algorithm.lbp(image_left, image_right, prior, 1.0, 10)
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
                prior_indices = [np.searchsorted(prior_info['ts'], t_frame, side="left")
                                 for t_frame in self.frames_timestamps]
                priors = prior_info['priors'][prior_indices]
            else:
                priors = prior_info['priors']
            assert len(priors) == len(self.frames_left) == len(self.frames_right)

            pb = ProgressBar(n_frames, "Starting offline frame-based stereo matching with prior initialisation.")
            start_timer = time.time()
            for i, (left, right, prior) in enumerate(zip(self.frames_left, self.frames_right, priors)):
                # left = restoration.denoise_bilateral(left, multichannel=False)
                # right = restoration.denoise_bilateral(right, multichannel=False)
                left = filters.rank.enhance_contrast(left.astype(np.int32), morphology.disk(4))
                right = filters.rank.enhance_contrast(right.astype(np.int32), morphology.disk(4))
                self.run_one_frame(left, right, prior)
                plt.imsave('output/head_downsampled/left_{}.png'.format(i), left)
                plt.imsave('output/head_downsampled/right_{}.png'.format(i), right)
                plt.imsave('output/head_downsampled/prior_{}.png'.format(i), prior)
                plt.imsave('output/head_downsampled/result_{}.png'.format(i), self.depth_frames[i])
                pb.update()
            end_timer = time.time()
            pb.end()
        else:
            pb = ProgressBar(n_frames, "Starting offline frame-based stereo matching without prior initialisation.")
            start_timer = time.time()
            for i, (left, right) in enumerate(zip(self.frames_left, self.frames_right)):
                self.run_one_frame(left, right)
                plt.imsave('output/head_downsampled/left_{}.png'.format(i), left)
                plt.imsave('output/head_downsampled/right_{}.png'.format(i), right)
                plt.imsave('output/head_downsampled/result_{}.png'.format(i), self.depth_frames[i])
                pb.update()
            end_timer = time.time()
            pb.end()
        logger.info("Frame-based stereo matching took {}s per image pair on average.".format((end_timer - start_timer)
                                                                                             / n_frames))
