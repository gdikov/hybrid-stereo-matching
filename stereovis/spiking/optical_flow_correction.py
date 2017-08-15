import numpy as np
import logging
import time

from stereovis.spiking.algorithms import VelocityVectorField
from stereovis.utils.frames_io import split_frames_by_time
from spinn_utilities.progress_bar import ProgressBar

logger = logging.getLogger(__file__)


class OpticalFlowPixelCorrection(object):
    def __init__(self, resolution, reference_events, buffer_pivots=None, buffer_interval=50, algorithm='vvf'):
        self.resolution = resolution
        if algorithm == 'vvf':
            self.algorithm = VelocityVectorField(time_interval=buffer_interval,
                                                 neighbourhood_size=(3, 3),
                                                 rejection_threshold=0.005,
                                                 convergence_threshold=1e-5,
                                                 max_iter_steps=5,
                                                 min_num_events_in_timespace_interval=10)
            indices_frames, _ = split_frames_by_time(ts=reference_events[:, 0],
                                                     time_interval=buffer_interval,
                                                     pivots=buffer_pivots)
            self.reference_frames_indices = indices_frames
            self.reference_events = reference_events
        else:
            raise NotImplementedError("Only VRF is supported.")

    def compute_velocity_field(self, timespace_frame, assume_sorted=False):
        return self.algorithm.fit_velocity_field(timespace_frame, assume_sorted=assume_sorted,
                                                 concatenate_polarity_groups=False)

    def adjust(self, event_frames, prior_nondata_value=-1, time_arrow=+1):
        """
        Run the frame-based stereo matching on all frames and priors.

        Args:
            event_frames: ndarray, the events which will be adjusted from the pre-computed velocity field
            prior_nondata_value: int, a value of the non-data pixels in the event_frames
            time_arrow: int, can be +1 or -1 which determines whether the events should be adjusted in the
                direction of the flow or the opposite. The fist is `predictive` and the second `retrospective` mode.

        Returns:
            Corrected events according to the velocity direction.

        Notes:
            The shift amounts to at most one pixel.
        """
        n_frames = len(self.reference_frames_indices)
        assert len(event_frames) == n_frames, "Mismatching reference and unadjusted target " \
                                              "frames, {} and {} frames respecively".format(n_frames, len(event_frames))
        pb = ProgressBar(n_frames, "Starting velocity field estimation for prior adjustment.")
        start_timer = time.time()

        adjusted_frames = np.empty_like(event_frames, dtype=np.float32)
        for i, prior_frame in enumerate(event_frames):
            timespace_frame = self.reference_events[self.reference_frames_indices[i]]
            # compute (x, y) velocities for each camera event
            velocities = self.compute_velocity_field(timespace_frame, assume_sorted=False)

            # compute shift based on 8 directional compass
            shifts = np.asarray([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=np.int32)
            compute_shift = lambda x, y: shifts[int(np.floor(np.round(8 * np.arctan2(y, x) / (2 * np.pi)))) % 8] \
                                         if np.linalg.norm([x, y]) > 1. else np.array([0, 0])

            # project the velocities onto a 2D image plane which will be queried for a shift
            mask_positive = timespace_frame[:, 3] == 0
            # FIXME: the velocity_frame is not used. Bug. Fix in the eval.ipynb first, test there and fix then here too.
            velocity_frame = np.zeros(self.resolution[::-1] + (2,))
            velocity_frame[timespace_frame[mask_positive][:, 2].astype(np.int),
                           timespace_frame[mask_positive][:, 1].astype(np.int)] = velocities['positive']
            velocity_frame[timespace_frame[~mask_positive][:, 2].astype(np.int),
                           timespace_frame[~mask_positive][:, 1].astype(np.int)] = velocities['negative']

            adjusted_frame = np.ones_like(prior_frame) * prior_nondata_value
            # compute the corresponding shift for all detected disparity event_frames
            for row, col in np.argwhere(prior_frame >= 0):
                dcol, drow = time_arrow * compute_shift(col, row)
                # going up in the image is equivalent to decrementing the row number, hence the minus in row - drow
                if 0 <= col + dcol < self.resolution[0] and 0 <= row - drow < self.resolution[1]:
                    adjusted_frame[row - drow, col + dcol] = prior_frame[row, col]
            adjusted_frames[i, :, :] = adjusted_frame
            pb.update()
        end_timer = time.time()
        pb.end()
        logger.info("Prior adaptation took {}s per frame on average.".format((end_timer - start_timer) / n_frames))
        return adjusted_frames
