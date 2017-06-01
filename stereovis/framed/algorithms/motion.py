import numpy as np
from scipy.linalg import lstsq


class VelocityField:
    def __init__(self):
        self.delta_t = 1
        self.neighbourhood_size = (3, 3)
        self.rejection_threshold = 0.005
        self.convergence_threshold = 1e-5
        self.max_iter_steps = 3

    def _fit_velocity_field(self, events):
        velocities = np.zeros((events.shape[0], 2))   # x, y velocity components
        for event_id, e in enumerate(events):
            # find time-wise near events
            candidate_events = events[np.searchsorted(events[:, 0], e[0] - self.delta_t, side='right'):
                                      np.searchsorted(events[:, 0], e[0] + self.delta_t, side='right')]
            # find space-wise near event (according to the Manhattan distance metric)
            candidate_events = candidate_events[(np.abs(candidate_events[:, 1] - e[1]) <= self.neighbourhood_size[0])
                                                & (np.abs(candidate_events[:, 2] - e[2]) <= self.neighbourhood_size[1])]

            # perform a normalisation (min subtraction) and reverse to the initial domain at the end
            min_txy = np.min(candidate_events, axis=0)
            candidate_events = candidate_events - min_txy

            num_candidates = candidate_events.shape[0]
            if num_candidates < 3:
                # no need to fit plane in time-space if there are insufficient events
                continue

            positions = np.hstack([candidate_events[:, 1:], np.ones((num_candidates, 1))])
            times = candidate_events[:, 0]
            current_best_params = lstsq(positions, times)[0]

            n_iter = iter(xrange(self.max_iter_steps+1))
            epsilon = np.inf
            while next(n_iter) < self.max_iter_steps and epsilon > self.convergence_threshold:
                v1 = np.array([current_best_params[0], 1, 0])
                v2 = np.array([current_best_params[1], 0, 1])
                plane_normal = np.cross(v1, v2)
                accepted_events = np.where(np.abs(np.dot(candidate_events, plane_normal))
                                           <= self.rejection_threshold)[0]
                # accepted_events = np.where(np.abs(np.dot(positions, current_best_params) - times)
                #                            <= self.rejection_threshold)[0]
                if accepted_events.size < 3 or accepted_events.size == num_candidates:
                    # too few events to work with, interrupt fine-tuning and use the obtained result so far
                    break
                new_best_params = lstsq(positions[accepted_events], times[accepted_events])[0]
                epsilon = np.linalg.norm(current_best_params - new_best_params)
                current_best_params = new_best_params

            # set the amplitude and the x, y components of the velocity vector for the current event
            velocities[event_id, :] = current_best_params[:-1]

        return velocities, events[:, 1:3]

    def compute_optical_flow(self, events, assume_sorted=False):
        if not assume_sorted:
            events = events[np.argsort(events[:, 0])]

        # group events by polarity and strip the polarity column:
        positive_polarity_indices = events[:, 3] == 0
        stream_positive = events[positive_polarity_indices][:, :-1]
        stream_negative = events[~positive_polarity_indices][:, :-1]
        positive_velocities, positive_positions = self._fit_velocity_field(stream_positive)
        negative_velocities, negative_positions = self._fit_velocity_field(stream_negative)
        return negative_velocities, negative_positions





if __name__ == '__main__':
    vf = VelocityField()

    # events = np.array([[1, 2, 2, 0],
    #                    [1.2, 3, 2, 0],
    #                    [1.3, 3, 3, 0]])
    from utils.visualisation import plot_velocity_field
    from utils.spikes_io import load_spikes
    events = load_spikes("/Users/admin/Documents/University/TUM/Master_Info/SS17/IDP/SemiframelessStereoVision/"
                         "hybrid-stereo-matching/data/input/spikes/head.npz", as_spike_source_array=False,
                         resolution=(240, 180), dt_thresh=1)
    from utils.frames_io import generate_frames_from_spikes
    frames, _, time_ind = generate_frames_from_spikes(resolution=(240, 180),
                                                      xs=events['right'][:, 1],
                                                      ys=events['right'][:, 2],
                                                      ts=events['right'][:, 0],
                                                      zs=events['right'][:, 3],
                                                      time_interval=20,
                                                      pivots=[10000], non_pixel_value=-1,
                                                      return_time_indices=True)

    # import matplotlib.pyplot as plt
    # plt.imshow(frames[0])
    # plt.show()
    print('start vflow')
    vs, positions = vf.compute_optical_flow(events['right'][time_ind, :][0], assume_sorted=False)
    # small_arrows_ind = (np.abs(vs[:, 0]) < 100) & (np.abs(vs[:, 1]) < 100)
    # vs = vs[small_arrows_ind, :]
    # positions = positions[small_arrows_ind, :]
    plot_velocity_field({'xs': positions[:, 0], 'ys': positions[:, 1],
                         'vel_xs': vs[:, 0], 'vel_ys': vs[:, 1]}, frames[0])

