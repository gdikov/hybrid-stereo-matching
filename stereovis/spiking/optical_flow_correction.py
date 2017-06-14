import numpy as np
import logging
import time

from stereovis.spiking.algorithms import VelocityVectorField
from stereovis.utils.frames_io import generate_frames_from_spikes
from spinn_machine.utilities.progress_bar import ProgressBar

logger = logging.getLogger(__file__)


class OpticalFlowPixelCorrection(object):
    def __init__(self, resolution, reference_events=None, algorithm='vvf'):
        if algorithm == 'vvf':
            self.algorithm = VelocityVectorField()
            self.event_frames, _, self.time_ind = generate_frames_from_spikes(resolution=resolution,
                                                                              xs=reference_events[:, 1],
                                                                              ys=reference_events[:, 2],
                                                                              ts=reference_events[:, 0],
                                                                              zs=reference_events[:, 3],
                                                                              time_interval=50,
                                                                              pivots=None,
                                                                              non_pixel_value=-1,
                                                                              return_time_indices=True)
        else:
            raise NotImplementedError("Only VRF is supported.")

    def compute_velocity_field(self):
        n_frames = len(self.event_frames)
        pb = ProgressBar(n_frames, "Starting velocity field estimation for prior adjustment.")
        start_timer = time.time()
        # TODO: compute the whole events time-varying velocity field, not only per frame...
        for i, frame in enumerate(self.event_frames):
            vs = self.algorithm.fit_velocity_field(self.event_frames['right'][self.time_ind[i], :],
                                                   assume_sorted=False)
            pb.update()
        end_timer = time.time()
        pb.end()
        logger.info("Velocity field estimation took {}s for {} events.".format((end_timer - start_timer), n_frames))
        self.velocity_field = vs
        return vs

    def adjust(self, events):
        """
        Run the frame-based stereo matching on all frames and priors.

        Args:
            events: ndarray, the events which will be adjusted from the pre-computed velocity field

        Returns:

        """


