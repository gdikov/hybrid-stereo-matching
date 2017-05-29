from utils.spikes_io import load_spikes
from utils.frames_io import generate_frames_from_spikes
import numpy as np
import matplotlib.pyplot
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_spikes(spikes, output_dir='data/output', from_retina=False, offsets=(0, 0), resolution=(240, 180),
                frame_lenght=1000, pivots=None, color_range=(0, 10)):

    parsed_spikes = load_spikes(spikes, resolution=resolution, simulation_time=15000, timestep_unit='us',
                                dt_thresh=1, as_spike_source_array=False)
    if pivots is not None:
        if isinstance(pivots, str) and pivots.endswith('npy'):
            pivots = np.load(pivots) / 1000.
    if from_retina:
        frames, timestamps = generate_frames_from_spikes(resolution=resolution,
                                                         xs=parsed_spikes['left'][:, 1] + offsets[0],
                                                         ys=parsed_spikes['left'][:, 2] + offsets[1],
                                                         ts=parsed_spikes['left'][:, 0],
                                                         zs=parsed_spikes['left'][:, 3],
                                                         time_interval=frame_lenght,
                                                         pivots=pivots, non_pixel_value=-1)
    else:
        frames, timestamps = generate_frames_from_spikes(resolution=resolution,
                                                         xs=parsed_spikes['xs'] + offsets[0],
                                                         ys=parsed_spikes['ys'] + offsets[1],
                                                         ts=parsed_spikes['ts'],
                                                         zs=parsed_spikes['disps'],
                                                         time_interval=frame_lenght,
                                                         pivots=pivots, non_pixel_value=-1)
    for ts, img in zip(timestamps, frames):
        plt.imshow(img, interpolation='none', vmin=color_range[0], vmax=color_range[1])
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, '{}.png'.format(ts)))
        plt.gcf().clear()
