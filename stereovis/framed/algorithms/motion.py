import numpy as np
from scipy.linalg import lstsq


class VelocityField:
    """
    Implementation of the [1] "Event-Based Visual Flow, R. Benosman et. al., IEEE TNNLS, VOL. 25, NO. 2, 2014" algorithm
    for optical flow using event-based vision sensors. 
    
    Basic principle: fit a plane on a constrained 3D space-time interval around each event and estimate the x, y slope.
    """
    def __init__(self, time_interval=20, neighbourhood_size=(3, 3), rejection_threshold=0.005,
                 convergence_threshold=1e-5, max_iter_steps=5, min_num_events_in_timespace_interval=10):
        """
        Args:
            time_interval: int, forward and backward time span in milliseconds of the space-time interval
            neighbourhood_size: tuple of int, spatial extent of the space-time interval in pixels
            rejection_threshold: flaot, perpendicular distance from the fitted plane, 
                which determines whether an event belong to the local group or not.
            convergence_threshold: float, the minimum change in parameters, which determines 
                whether the rejection-fitting loop should continue or not
            max_iter_steps: int, maximum iteration steps, which is a complementary limitation of the loop.
            min_num_events_in_timespace_interval: int, the number of events which are sufficient for the plane fitting.
        """
        self.delta_t = time_interval
        self.neighbourhood_size = neighbourhood_size
        self.rejection_threshold = rejection_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iter_steps = max_iter_steps
        self.min_num_events_in_timespace_interval = min_num_events_in_timespace_interval

    def _fit_velocity_field(self, events):
        """
        Implement the velocity field fitting algorithm as described by [1]. 
        
        Args:
            events: ndarray of shape Nx3 with N points, each containing a timestamp and x, y coordinates. 

        Returns:
            A Nx2 ndarray with the x, y velocity of each event from the input. 
        """
        velocities = np.zeros((events.shape[0], 2))   # x, y velocity components
        for event_id, e in enumerate(events):
            # find time-wise near events
            candidate_events = events[np.searchsorted(events[:, 0], e[0] - self.delta_t, side='right'):
                                      np.searchsorted(events[:, 0], e[0] + self.delta_t, side='right')]
            # find space-wise near event (according to the Manhattan distance metric)
            candidate_events = candidate_events[(np.abs(candidate_events[:, 1] - e[1]) <= self.neighbourhood_size[0])
                                                & (np.abs(candidate_events[:, 2] - e[2]) <= self.neighbourhood_size[1])]

            num_candidates = candidate_events.shape[0]
            if num_candidates < self.min_num_events_in_timespace_interval:
                # no need to fit plane in time-space if there are insufficient events
                continue

            positions = np.hstack([candidate_events[:, 1:], np.ones((num_candidates, 1))])
            times = candidate_events[:, 0]
            current_best_params = lstsq(positions, times)[0]

            n_iter = iter(xrange(self.max_iter_steps+1))
            epsilon = np.inf
            while next(n_iter) < self.max_iter_steps and epsilon > self.convergence_threshold:
                # v1 = np.array([current_best_params[0], 1, 0])
                # v2 = np.array([current_best_params[1], 0, 1])
                plane_normal = np.hstack((current_best_params[2], current_best_params[:-1]))#np.cross(v1, v2)
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                accepted_events = np.where(np.abs(np.dot(candidate_events, plane_normal))
                                           <= self.rejection_threshold)[0]
                # accepted_events = np.where(np.abs(np.dot(positions, current_best_params) - times)
                #                            <= self.rejection_threshold)[0]
                if accepted_events.size < self.min_num_events_in_timespace_interval \
                        or accepted_events.size == num_candidates:
                    # too few events to work with, interrupt fine-tuning and use the obtained result so far
                    break
                new_best_params = lstsq(positions[accepted_events], times[accepted_events])[0]
                epsilon = np.linalg.norm(current_best_params - new_best_params)
                current_best_params = new_best_params

            # set the amplitude and the x, y components of the velocity vector for the current event
            velocities[event_id, :] = current_best_params[:-1]

        return velocities

    def fit_velocity_field(self, events, assume_sorted=False):
        """
        Compute velocity field for events. Group events by polarity and fit the velocity field on each group. 
        Merge the two results into a single one.
        
        Args:
            events: ndarray, Nx4 array with N events, each with a timestamp, x, y coordinates and a polarity bit.
            assume_sorted: bool, whether the events are sorted by time.

        Returns:
            A Nx2 numpy array with the velocity in x and y direction for each event from the input. 
        """
        if not assume_sorted:
            events = events[np.argsort(events[:, 0])]

        # group events by polarity and strip the polarity column:
        positive_polarity_indices = events[:, 3] == 0
        stream_positive = events[positive_polarity_indices][:, :-1]
        stream_negative = events[~positive_polarity_indices][:, :-1]
        positive_velocities = self._fit_velocity_field(stream_positive)
        negative_velocities = self._fit_velocity_field(stream_negative)
        all_velocities = np.vstack([positive_velocities, negative_velocities])
        return all_velocities

# if __name__ == '__main__':
#     vf = VelocityField()
#
#     # events = np.array([[1, 2, 2, 0],
#     #                    [1.2, 3, 2, 0],
#     #                    [1.3, 3, 3, 0]])
#     from utils.visualisation import plot_optical_flow
#     from utils.spikes_io import load_spikes
#     events = load_spikes("/Users/admin/Documents/University/TUM/Master_Info/SS17/IDP/SemiframelessStereoVision/"
#                          "hybrid-stereo-matching/data/input/spikes/head.npz", as_spike_source_array=False,
#                          resolution=(240, 180), dt_thresh=1)
#     from utils.frames_io import generate_frames_from_spikes
#     frames, _, time_ind = generate_frames_from_spikes(resolution=(240, 180),
#                                                       xs=events['right'][:, 1],
#                                                       ys=events['right'][:, 2],
#                                                       ts=events['right'][:, 0],
#                                                       zs=events['right'][:, 3],
#                                                       time_interval=50,
#                                                       pivots=list(range(9000, 13000, 100)), non_pixel_value=-1,
#                                                       return_time_indices=True)
#
#     # import matplotlib.pyplot as plt
#     # plt.imshow(frames[0])
#     # plt.show()
#     print('start vflow')
#     for i, frame in enumerate(frames):
#         vs = vf.fit_velocity_field(events['right'][time_ind[i], :], assume_sorted=False)
#         # small_arrows_ind = (np.abs(vs[:, 0]) < 100) & (np.abs(vs[:, 1]) < 100)
#         # vs = vs[small_arrows_ind, :]
#         # positions = positions[small_arrows_ind, :]
#         fig = plot_optical_flow({'xs': events['right'][time_ind[i], 1], 'ys': events['right'][time_ind[i], 1:3][:, 2],
#                                  'vel_xs': vs[:, 0], 'vel_ys': vs[:, 1]}, frame)
#         fig.savefig('../../../data/output/{:04d}.png'.format(i))
#         fig.clf()
