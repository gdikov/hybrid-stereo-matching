from utils.frames_io import generate_frames_from_spikes
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os


def plot_spikes(spikes, output_dir='data/output', image_naming='timestamps', resolution=(240, 180),
                offsets=(0, 0), frame_length=1000, pivots=None, color_range=(0, 10)):
    """
    Plot spikes as frames, exported as png images.
    
    Args:
        spikes: dict, the spikes from the retina or the network with given `xs`, `ys` for all the x and y coordinates,
            `ts` for the timestamps and `zs` for the spike value (e.g. disparity or polarity)
        output_dir: str, directory where the exported frames should be stored
        image_naming: str, can be `timestamps` or `enumerate` for the naming of the exported images
        resolution: tuple, the resolution of the exported images in (x,y) pixels
        offsets: tuple, if the resolution of the image is larger then the spikes' then an (x,y) offset will center it 
        frame_length: int, time in ms for the buffering of events into frames
        pivots: the time steps at which the frames will be created (taking events from the last `frame_length` ms) 
        color_range: tuple, the min und max disparity/pixel value, needed to adjust the colorbar

    Returns:
        
    """

    if pivots is not None:
        if isinstance(pivots, str) and pivots.endswith('npy'):
            pivots = np.load(pivots) / 1000.

    frames, timestamps = generate_frames_from_spikes(resolution=resolution,
                                                     xs=spikes['xs'] + offsets[0],
                                                     ys=spikes['ys'] + offsets[1],
                                                     ts=spikes['ts'],
                                                     zs=spikes['zs'],
                                                     time_interval=frame_length,
                                                     pivots=pivots, non_pixel_value=-1)
    for i, (ts, img) in enumerate(zip(timestamps, frames)):
        plt.imshow(img, interpolation='none', vmin=color_range[0], vmax=color_range[1])
        plt.colorbar()
        plt.savefig(os.path.join(output_dir,
                                 '{:04d}.png'.format(i) if image_naming == 'enumerate' else '{}.png'.format(ts)))
        plt.gcf().clear()


def plot_optical_flow(velocities, background=None):
    xs, ys, us, vs = velocities['xs'], velocities['ys'], velocities['vel_xs'], velocities['vel_ys']
    plt.figure()
    if background is not None:
        plt.imshow(background)

    ax = plt.gca()

    colors = np.arctan2(us, vs)

    norm = Normalize()
    if colors.size > 0:
        norm.autoscale(colors)
    colormap = cm.inferno
    ax.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=1, color=colormap(norm(colors)))
    plt.draw()
    plt.show()
