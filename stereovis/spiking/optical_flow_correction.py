import numpy as np
import logging
import time

from stereovis.spiking.algorithms import VelocityVectorField
from stereovis.utils.frames_io import split_frames_by_time, generate_frames_from_spikes
from spinn_machine.utilities.progress_bar import ProgressBar

logger = logging.getLogger(__file__)


class OpticalFlowPixelCorrection(object):
    def __init__(self, resolution, reference_events, buffer_pivots=None, buffer_interval=50, algorithm='vvf'):
        self.resolution = resolution
        if algorithm == 'vvf':
            self.algorithm = VelocityVectorField()
            indices_frames, _ = split_frames_by_time(ts=reference_events[:, 0],
                                                     time_interval=buffer_interval,
                                                     pivots=buffer_pivots)
            self.reference_frames = np.split(reference_events, indices_frames)
        else:
            raise NotImplementedError("Only VRF is supported.")

    def compute_velocity_field(self, timespace_frame, assume_sorted=False):
        return self.algorithm.fit_velocity_field(timespace_frame, assume_sorted=assume_sorted,
                                                 concatenate_polarity_groups=False)

    def adjust(self, events):
        """
        Run the frame-based stereo matching on all frames and priors.

        Args:
            events: ndarray, the events which will be adjusted from the pre-computed velocity field

        Returns:
            Corrected events according to the velocity direction.

        Notes:
            The shift amounts to at most one pixel.
        """
        n_frames = len(self.reference_frames)
        assert len(events) == n_frames, "Mismatching reference and unadjusted target " \
                                        "frames, {} and {} frames respecively".format(n_frames, len(events))
        pb = ProgressBar(n_frames, "Starting velocity field estimation for prior adjustment.")
        start_timer = time.time()

        adjusted_events = events.copy()
        for i, (timespace_frame, prior_frame) in enumerate(zip(self.reference_frames, events)):
            # compute (x, y) velocities for each camera event
            velocities = self.compute_velocity_field(timespace_frame, assume_sorted=False)

            # compute shift based on 8 directional compass
            shifts = np.asarray([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
            compute_shift = lambda x, y: shifts[int(np.floor(np.round(8 * np.arctan2(y, x) / (2*np.pi)))) % 8] \
                                         if np.linalg.norm([x, y]) > 1. else np.array([0, 0])

            # project the velocities onto a 2D image plane which will be queried for a shift
            mask_positive = timespace_frame[:, 3] == 0
            velocity_frame = np.zeros(self.resolution[::-1])
            velocity_frame[timespace_frame[mask_positive][:, 2],
                           timespace_frame[mask_positive][:, 1]] = velocities['positive']
            velocity_frame[timespace_frame[~mask_positive][:, 2],
                           timespace_frame[~mask_positive][:, 1]] = velocities['negative']

            # compute the corresponding shift for all events
            # FIXME: compute the shift of the arrow relative to the pixel, not to the image center. verify the velocity
            # values
            for j, e in enumerate(prior_frame):
                adjusted_events[i, j, 1:3] += compute_shift(e[1], e[2])
            pb.update()
        end_timer = time.time()
        pb.end()
        logger.info("Prior adaptation took {}s for {} events.".format((end_timer - start_timer), n_frames))
        return adjusted_events



