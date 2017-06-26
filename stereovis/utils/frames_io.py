import re
import numpy as np
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime as dt
from skimage.io import imread
from skimage.transform import rescale

logger = logging.getLogger(__file__)


def load_ground_truth(filename):
    """
    Load the disparity ground truth for some sample images. 
    
    Args:
        filename: a PGM or PFM file path.

    Returns:
        A numpy array of integers.
    """
    fp = open(filename, 'rb')
    if filename.endswith('.pgm'):
        assert fp.readline() == 'P5\n'
        (width, height) = [int(i) for i in fp.readline().split()]
        depth = int(fp.readline())
        assert depth <= 255
        raster = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append(ord(fp.read(1)))
            raster.append(row)
        return np.asarray(raster) / 14

    elif filename.endswith('.pfm'):
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(fp.readline().rstrip())
        # little endian
        if scale < 0:
            endian = '<'
        # big endian
        else:
            endian = '>'

        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data[data == np.inf] = 0
        return np.flipud(np.reshape(data, shape))
    else:
        raise ValueError("Unknown file type")


def load_frames(input_path, crop_region=None, resolution=None, scale_down_factor=(1, 1),
                simulation_time=None, timestamp_unit='us'):
    """
    Load the input frames, sorted by time and possibly cropped to some resolution in some region. 
    
    Args:
        input_path: the path to the images or a path to a single file (all in png format)
        crop_region: top left corner for the bounding box of the region of interest
        resolution: the shape of the image to be read
        scale_down_factor: tuple of the x and y down-sampling factors
        simulation_time: the start and stop time or stop time of the simulation, used to take the frames of interest.
        timestamp_unit: the units of time for the timestamps. Can be `us` or `ms`.

    Returns:
        A numpy array of shape KxNxM where N is the number of loaded images of dimension NxM and a list of timestamps
    """
    if crop_region is not None and resolution is not None:
        # top left coordinates in x, y order
        col_start, row_start = crop_region
        dx, dy = resolution
        col_stop, row_stop = col_start + dx, row_start + dy
    else:
        col_start, col_stop, row_start, row_stop = 0, 0, 0, 0
    scale_down_factor = np.asarray(scale_down_factor)

    def downsample(image_to_scale):
        rescaled_img = rescale(image_to_scale, 1.0 / scale_down_factor, preserve_range=True)
        return rescaled_img

    if os.path.isdir(input_path):
        if isinstance(simulation_time, tuple):
            min_t, max_t = simulation_time
        elif isinstance(simulation_time, int):
            min_t, max_t = 0, simulation_time
        else:
            min_t, max_t = 0, np.iinfo(np.int32).max

        # since image names encode the timestamps sorting by them is equivalent to sorting by time
        image_files = [os.path.join(input_path, str(i) + '.png') for i in
                       sorted([int(os.path.splitext(os.path.basename(img))[0]) for img in os.listdir(input_path)
                               if os.path.splitext(os.path.basename(img))[1] == '.png'])
                       if min_t <= i <= max_t]

        if not (col_stop or col_start or row_stop or row_start):
            images = np.stack([downsample(imread(img)) for img in image_files])
        else:
            images = np.stack([downsample(imread(img)[row_start:row_stop, col_start:col_stop]) for img in image_files])

        # load the timestamps if available
        timestamps_file = os.path.join(input_path, 'timestamps.npy')
        if os.path.exists(timestamps_file):
            timestamps = np.load(timestamps_file)
            if timestamp_unit == 'us':
                timestamps = timestamps / 1000.0
        else:
            logger.error("No timestamps file located in the frames folder!")
            timestamps = None
        return images, timestamps

    else:
        if not (col_stop or col_start or row_stop or row_start):
            return downsample(imread(input_path)), None
        else:
            return downsample(imread(input_path)[row_start:row_stop, col_start:col_stop]), None


def save_frames(frames, output_dir):
    """
    Save the frames as png images.
    
    Args:
        frames: a list of frames (numpy arrays) to be saved 
        output_dir: a path to the output directory 

    Returns:
        In-place method.
    """
    timestamp = str(dt.now().isoformat())
    full_output_dir = os.path.join(output_dir, "depth-frames_" + timestamp)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    logger.info("Saving frames to {}".format(full_output_dir))
    min_val, max_val = np.min(frames), np.max(frames)
    cmap = plt.get_cmap('jet', int(max_val))
    cmap.set_under('gray')
    cmap.set_bad('gray')
    for i, frame in enumerate(frames):
        # frame = ((frame / float(max_val) - min_val) * 255).astype(np.int8)
        # imsave(os.path.join(full_output_dir, '{}.png'.format(i)), frame)
        plt.imshow(frame, interpolation='none', vmin=min_val+0.1, vmax=max_val, cmap=cmap)
        plt.colorbar()
        plt.savefig(os.path.join(full_output_dir, '{}.png'.format(i)))
        plt.gcf().clear()


def split_frames_by_time(ts, start_time=0, time_interval=100, pivots=None):
    """

    Args:
        ts:
        start_time:
        time_interval:
        pivots:

    Returns:

    """
    ts = np.asarray(ts)
    # sort the spike times by time and use the sorted indices order to access the events in chronological order too
    sorted_indices = np.argsort(ts)
    sorted_time = ts[sorted_indices]
    # remove the events which happened before the `start_time`
    sorted_time = sorted_time[sorted_time >= start_time]
    sorted_indices = sorted_indices[sorted_indices.size - sorted_time.size:]

    if pivots is None:
        # compute the differences in spiking times and split the whole event stream into time-wise equally sized frames
        frame_ticks_indices = np.convolve(sorted_time % time_interval, [1, -1], mode='valid') < 0
        frame_ticks_indices = np.where(np.concatenate([np.array([False]), frame_ticks_indices]))[0]
        timestamps = sorted_time[frame_ticks_indices]
        # clip the end since there is no timestamp information for it and the frame may be incomplete.
        indices_frames = np.split(sorted_indices, frame_ticks_indices)[:-1]
    else:
        indices_frames = [sorted_indices[np.where(np.logical_and(sorted_time >= tick - time_interval,
                                                                 sorted_time <= tick))[0]] for tick in pivots]
        timestamps = pivots
    return indices_frames, timestamps


def generate_frames_from_spikes(resolution, xs, ys, ts, zs, start_time=0, time_interval=100,
                                pivots=None, non_pixel_value='nan', return_time_indices=False):
    """
    Generate frames from spikes given `x`, `y` coordinates, timestamp (`t`) for all spikes and possibly a 
    `z` fill value for the pixel.
    
    Args:
        resolution: the resolution of the frames -- a tuple of x and y dimensionality
        xs: a list of x-coordinates for each spike
        ys: a list of y-coordinates for each spike
        ts: a list ot timestamps for each spike
        zs: a list of fill values
        start_time: optional, a starting time for the frames
        time_interval: optional, the time length of the buffering for each frame in milliseconds
        pivots: optional, a list of time ticks at which the frames are built (buffered `time_interval` ms before)
        non_pixel_value: the value on the pixels where there is no data available
        return_time_indices: bool, whether the indices of the timestamps at which frames are created should be returned 

    Returns:
        A numpy array of shape N x *`resolution` representing the buffered frames.
    """
    logger.info("Generating {} frames from spikes".format(len(pivots) if pivots is not None
                                                          else int(np.max(ts)/time_interval)))
    xs, ys, ts, zs = np.asarray(xs).astype(np.int), np.asarray(ys).astype(np.int), np.asarray(ts), np.asarray(zs)

    indices_frames, timestamps = split_frames_by_time(ts, start_time, time_interval, pivots)

    # buffer the events into frames
    frames_count = len(indices_frames)
    frames_cols, frames_rows, frames_vals = zip(*[(xs[inds], ys[inds], zs[inds]) for inds in indices_frames])
    n_cols, n_rows = resolution
    non_pixel_value = float(non_pixel_value)
    frames = np.ones((frames_count, n_rows, n_cols)) * non_pixel_value
    for i, (cols, rows, vals) in enumerate(zip(frames_cols, frames_rows, frames_vals)):
        if cols.size == 0:
            # empty frame, skip value setting
            continue
        frames[i, rows, cols] = vals

    if return_time_indices:
        return frames, timestamps, indices_frames
    return frames, timestamps
