import re
import numpy as np
import os
from skimage.io import imread


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
        for y in range(height):
            row = []
            for y in range(width):
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
            scale = -scale
        # big endian
        else:
            endian = '>'

        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data[data == np.inf] = 0
        return np.flipud(np.reshape(data, shape))
    else:
        raise ValueError("Unknown file type")


def load_frames(input_path, crop_region=None, resolution=None, simulation_time=None):
    """
    Load the input frames, sorted by time and possibly cropped to some resolution in some region. 
    
    Args:
        input_path: the path to the images or a path to a single file (all in png format)
        crop_region: top left corner for the bounding box of the region of interest
        resolution: the shape of the image to be read
        simulation_time: the start and stop time or stop time of the simulation, used to take the frames of interest.

    Returns:
        A numpy array of shape KxNxM where N is the number of loaded images of dimension NxM
    """
    if crop_region is not None and resolution is not None:
        # top left coordinates in x, y order
        col_start, row_start = crop_region
        dx, dy = resolution
        col_stop, row_stop = col_start + dx, row_start + dy
    else:
        col_start, col_stop, row_start, row_stop = 0, 0, 0, 0

    if os.path.isdir(input_path):
        if isinstance(simulation_time, tuple):
            min_t, max_t = simulation_time
        elif isinstance(simulation_time, int):
            min_t, max_t = 0, simulation_time
        else:
            min_t, max_t = 0, np.iinfo(np.int64).max

        # since image names encode the timestamps sorting by them is equivalent to sorting by time
        image_files = [os.path.join(input_path, str(i) + '.png') for i in
                       sorted([int(os.path.splitext(os.path.basename(img))[0]) for img in os.listdir(input_path)
                               if os.path.splitext(os.path.basename(img))[1] == '.png'])
                       if min_t <= i <= max_t]

        if not col_stop or col_start or row_stop or row_start:
            images = np.stack([imread(img) for img in image_files])
        else:
            images = np.stack([imread(img)[row_start:row_stop, col_start:col_stop] for img in image_files])
        return images

    else:
        if col_stop or col_start or row_stop or row_start:
            return imread(input_path)
        else:
            return imread(input_path)[row_start:row_stop, col_start:col_stop]


def save_frames():
    raise NotImplementedError


def generate_frames_from_spikes(resolution, xs, ys, ts, zs=None, start_time=0, time_interval=100):
    """
    Generate frames from spikes given `x`, `y` coordinates, timestamp (`t`) for all spikes and possibly a 
    `z` fill value for the pixel.
    
    Args:
        resolution: the resolution of the frames -- a tuple of x and y dimensionality
        xs: a list of x-coordinates for each spike
        ys: a list of y-coordinates for each spike
        ts: a list ot timestamps for each spike
        zs: optional, a list of fill values
        start_time: optional, a starting time for the frames
        time_interval: the time length of the buffering for each frame in milliseconds

    Returns:
        A numpy array of shape N x *`resolution` representing the buffered frames.
    """
    # sort the spike times by time and use the sorted indices order to access the events in chronological order too
    sorted_indices = np.argsort(ts)
    sorted_time = ts[sorted_indices]
    # remove the events which happened before the `start_time`
    sorted_time = sorted_time[sorted_time >= start_time]
    sorted_indices = sorted_indices[len(sorted_indices) - len(sorted_time):]
    # compute the differences in spiking times and split the whole event stream into time-wise equally sized frames
    delta_ts = np.convolve(sorted_time, [1, -1], mode='valid')
    indices_accumulated_frames = np.concatenate([np.array([False]), np.cumsum(delta_ts) % (time_interval + 1) == 0])
    indices_frames = np.split(sorted_indices, sorted_indices[indices_accumulated_frames])
    indices_frames = indices_frames[:-1] if indices_frames[-1].shape == indices_frames[0].shape else indices_frames
    indices_frames = np.array(indices_frames)
    # buffer the events into frames
    frames_count = len(indices_frames)
    frames_xs, frames_ys, frames_zs = xs[indices_frames], ys[indices_frames], zs[indices_frames]
    n_cols, n_rows = resolution
    frames = np.zeros((frames_count, n_rows, n_cols), dtype=np.int32)
    as_binary = zs is not None
    for i, x, y, z in enumerate(zip(frames_xs, frames_ys, frames_zs)):
        # set pixel to 1 if spike has occurred at any time during the frame
        if as_binary:
            frames[i, x, y] = 1
        # otherwise, set with the intensity value
        else:
            frames[i, x, y] = z

    return frames
