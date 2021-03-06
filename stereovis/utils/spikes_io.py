import urllib
import re
import contextlib
import os
import numpy as np
import cPickle
from datetime import datetime as dt

import logging

logger = logging.getLogger(__file__)


class SpikeParser:
    def __init__(self, crop_region=None, resolution=None, simulation_time=None,
                 timestep_unit='us', scale_down_factor=(1, 1)):
        """
        Args:
            crop_region: a tuple of top left and bottom right pixel coordinates, which will be cropped from the 
             left and right retinas respectively. 
            resolution: a tuple of retina's x and y resolution (pixel count)
            simulation_time: a tuple with start and end time or an int with an end time
            timestep_unit: str, the unit in which the events are recorded
            scale_down_factor: tuple, the downsampling factor in x and y direction.
        """
        # define the crop region by a top left and bottom right corner coordinates
        dx, dy = resolution
        if crop_region is not None:
            self.crop_region = crop_region + (crop_region[0] + dx - 1, crop_region[1] + dy - 1)
        else:
            self.crop_region = (0, 0) + (dx - 1, dy - 1)
        self.effective_resolution = resolution

        if simulation_time is not None:
            if isinstance(simulation_time, tuple):
                self.simulation_start_time, self.simulation_end_time = simulation_time
            elif isinstance(simulation_time, int):
                self.simulation_start_time, self.simulation_end_time = 0, simulation_time
            else:
                raise TypeError("`simulation_time` is expected to be an int denoting the end time "
                                "or a tuple with start and end times.")
        else:
            self.simulation_end_time = None
            self.simulation_start_time = 0

        self.time_unit = timestep_unit
        self.scale_down_factor = scale_down_factor

    def parse(self, input_data):
        """
        Parse the input data. It can be a filename or a url to a '.dat' or '.txt' file. If it is a local filename,
        numpy compressed or numpy pickles are also parsed.
        
        Args:
            input_data: a path to a 'npz', 'dat' or 'txt' file or a url to raw text ('dat' or 'txt')

        Returns:
            A dict with keys 'left' and 'right' for the two views, with numpy arrays of shape Nx4 and N'x4.
            The four columns are for the timestamp, x, y coordinates and polarity (as in the input files).
        """
        data_url_regex = re.compile(r'^(?:http|ftp)s?://'  # http:// or https://
                                    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
                                    r'localhost|'  # localhost...
                                    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                                    r'(?::\d+)?'  # optional port
                                    r'(?:/?|[/?]\S+)(:?.dat|.txt)$$', re.IGNORECASE)

        is_input_url = re.match(data_url_regex, input_data) is not None

        def raw2num(data):
            events = {'left': [], 'right': []}
            for event in data.split('\n'):
                event = map(int, event.split())
                # interpret 0 as right retina id and discard the retina id flag when storing events
                if len(event) == 5 and event[-1] == 1:
                    events['left'].append(event[:-1])
                elif len(event) == 5 and event[-1] == 0:
                    events['right'].append(event[:-1])
            events['left'] = np.asarray(events['left']).astype(np.float)
            events['right'] = np.asarray(events['right']).astype(np.float)
            return events

        if is_input_url:
            logger.debug("Parsing online spike file.")
            # connect to website and parse text data
            with contextlib.closing(urllib.urlopen(input_data)) as fp:
                raw_data = fp.read()
                raw_data = raw2num(raw_data)
        else:
            logger.debug("Parsing local spike file.")
            filename = os.path.basename(input_data)
            if filename.endswith('dat') or filename.endswith('txt'):
                with open(input_data, 'r') as fp:
                    raw_data = fp.read()
                    raw_data = raw2num(raw_data)
            elif filename.endswith('npz') or filename.endswith('npy'):
                # a dict with 'left' and 'right' keys containing a numpy arrays of N spikes
                # each of which is represented by a timestamp, x, y coordinate and a polarity value.
                file_data = np.load(input_data)
                if 'left' in file_data.keys() and 'right' in file_data.keys():
                    raw_data = {'left': np.asarray(file_data['left']).astype(np.float),
                                'right': np.asarray(file_data['right']).astype(np.float)}
                else:
                    raise ValueError("Expected a npz file with keys `left` and `right`.")
            else:
                ValueError("Unknown file type. Provide a .txt, .dat, .npz file or url.")

        # needs to be sorted by timestamp
        logger.debug("Sorting file by spike times.")
        raw_data['left'] = raw_data['left'][raw_data['left'][:, 0].argsort()]
        raw_data['right'] = raw_data['right'][raw_data['right'][:, 0].argsort()]
        return raw_data

    def sanitise_events(self, events, dt_threshold, assume_sorted=True):
        """
        Sanitise raw event data: clear event bursts and crop from the region of interest.
        
        Args:
            events: a dict with the raw events. Keys 'left' and 'right' contain Nx4 and N'x4 numpy arrays
            dt_threshold: int with the shortest time difference between two consecutive events
            assume_sorted: bool whether the events are sorted by time

        Returns:
            the same dict as the input raw events but with the inappropriate events filtered out.
        """
        logger.debug("Performing region cropping on the events.")
        # filter events which lie outside the crop_region and normalise the x, y coordinates to [0, n-1]
        events['left'] = events['left'][(self.crop_region[0] <= events['left'][:, 1])
                                        & (events['left'][:, 1] <= self.crop_region[2])
                                        & (self.crop_region[1] <= events['left'][:, 2])
                                        & (events['left'][:, 2] <= self.crop_region[3])]
        events['left'][:, 1:3] -= np.array((self.crop_region[0], self.crop_region[1]))
        events['right'] = events['right'][(self.crop_region[0] <= events['right'][:, 1])
                                          & (events['right'][:, 1] <= self.crop_region[2])
                                          & (self.crop_region[1] <= events['right'][:, 2])
                                          & (events['right'][:, 2] <= self.crop_region[3])]
        events['right'][:, 1:3] -= np.array((self.crop_region[0], self.crop_region[1]))

        # convert time to ms
        if self.time_unit == 'us':
            logger.debug("Converting from us to ms time units. This may introduce rounding errors.")
            events['left'][:, 0] /= 1000.0
            events['right'][:, 0] /= 1000.0

        if not assume_sorted:
            logger.debug("Sorting events by spike times.")
            events['left'] = events['left'][events['left'][:, 0].argsort()]
            events['right'] = events['right'][events['right'][:, 0].argsort()]

        # get the time region of interest
        if self.simulation_end_time is not None:
            logger.debug("Performing time cropping on the events.")
            end_index_left = np.argmax(events['left'][:, 0] > self.simulation_end_time)
            end_index_left = len(events['left'][:, 0]) if end_index_left == 0 else end_index_left
            events['left'] = events['left'][np.argmax(events['left'][:, 0] >= self.simulation_start_time):
                                            end_index_left, :]
            end_index_right = np.argmax(events['left'][:, 0] > self.simulation_end_time)
            end_index_right = len(events['left'][:, 0]) if end_index_right == 0 else end_index_right
            events['right'] = events['right'][np.argmax(events['right'][:, 0] >= self.simulation_start_time):
                                              end_index_right, :]

        def apply_time_filter(event_list):
            last_spikes = -2 * dt_threshold * np.ones(self.effective_resolution)
            allowed_indices = []
            for i, (t, x, y, _) in enumerate(event_list):
                x, y = int(x), int(y)
                if t - last_spikes[x, y] > dt_threshold:
                    allowed_indices.append(i)
                last_spikes[x, y] = t
            return event_list[allowed_indices]

        # down-sample the events first and then filter the bursts, as this operation increases the event density.
        # otherwise the dt_threshold constraint might be unsatisfied.
        if self.scale_down_factor != (1, 1):
            logger.debug("Performing downscaling.")
            events['left'] = self.downsample(events['left'])
            events['right'] = self.downsample(events['right'])
            self.effective_resolution = np.asarray(self.effective_resolution) / np.asarray(self.scale_down_factor)

        # filter event bursts, i.e. spikes which occur faster than a dt_threshold
        logger.debug("Filtering event bursts within time interval of {}ms.".format(dt_threshold))
        events['left'] = apply_time_filter(events['left'])
        events['right'] = apply_time_filter(events['right'])

        return events

    def downsample(self, events, mode='skip'):
        """
        Down-sample the resolution of the events.
         
        Args:
            events: a numpy array with the events containing a timestamp, x, y coordinate and polarity.
            mode: str, the way events are down-sampled. Can be `aggregate` for combining adjacent pixel to 
                map to the same down-sampled one, or `skip` to skip all but the n-th pixel. The latter results in more 
                sparse and naturally looking result.

        Returns:
            The modified, down-sampled events.
        """
        if mode == 'aggregate':
            events[:, 1:3] /= np.asarray(self.scale_down_factor)
        elif mode == 'skip':
            events = events[(events[:, 1] % self.scale_down_factor[0] == 0)
                            & (events[:, 2] % self.scale_down_factor[1] == 0)]
            events[:, 1:3] /= np.asarray(self.scale_down_factor)
        return events


def load_spikes(input_file, crop_region=None, resolution=None,
                simulation_time=None, timestep_unit='us', dt_thresh=0,
                scale_down_factor=(1, 1), as_spike_source_array=False):
    """
    Load the spikes form a file or a url into a list of populations each with lists of neuron spiking times or 
    as a list of individual spikes. 
     
    Args:
        input_file: path to input data file or a url
        crop_region: the region of interest in pixels defined by top left coordinates in x, y order
        resolution: the standard retina resolution
        simulation_time: simulation start and end time or simulation end time only
        timestep_unit: the units in which the timestamps are encoded. Can be 'us' or 'ms'.
        dt_thresh: shortest amount of time in which same pixels are not allowed to spike. 
        scale_down_factor: tuple, the down-sampling factor for the resolution.
        as_spike_source_array: bool flag whether the loaded spikes should be packed into SpikeSourceArray-friendly 
            object (list of lists of lists) or just as-is (list of spikes with timestamp, x, y, polarity)

    Returns:
        A dict with 'left' and 'right' retina spiking times.
    """
    logger.info("Loading spikes from {}".format(input_file))
    if os.path.splitext(os.path.basename(input_file))[1] == '.pickle':
        try:
            with open(input_file, 'rb') as fp:
                spikes = cPickle.load(fp)
        except IOError:
            logger.error("No network output spikes found. Run the spiking network first.")
            spikes = None
        return spikes

    parser = SpikeParser(crop_region=crop_region, resolution=resolution, simulation_time=simulation_time,
                         timestep_unit=timestep_unit, scale_down_factor=scale_down_factor)
    logger.debug("Effective retina resolution after pre-processing is {}".format(parser.effective_resolution))
    raw_data = parser.parse(input_file)
    filtered_data = raw_data
    if crop_region is not None or simulation_time is not None or dt_thresh > 0:
        filtered_data = parser.sanitise_events(raw_data, dt_thresh, assume_sorted=True)
    if not as_spike_source_array:
        return filtered_data
    logger.debug("Creating SpikeSourceArray spike-times objects.")
    # create lists of (populations) lists of (neuron's spiking times) lists and fill in the time values
    max_t = np.max([np.max(filtered_data['left'][:, 0]), np.max(filtered_data['right'][:, 0])]) + 10
    n_cols, n_rows = parser.effective_resolution
    retina_spikes = {'left': [[[] for _ in xrange(n_rows)] for _ in xrange(n_cols)],
                     'right': [[[] for _ in xrange(n_rows)] for _ in xrange(n_cols)]}
    for t, population_id, neuron_id, _ in filtered_data['left']:
        retina_spikes['left'][int(population_id)][int(neuron_id)].append(t)
    for t, population_id, neuron_id, _ in filtered_data['right']:
        retina_spikes['right'][int(population_id)][int(neuron_id)].append(t)
    # append a fictitious spiking time which is never reached.
    # It is necessary for the SpikeSourceArray requires a value for each neuron's spiking
    for population_id in xrange(n_cols):
        for neuron_id in xrange(n_rows):
            retina_spikes['left'][population_id][neuron_id].append(max_t)
            retina_spikes['right'][population_id][neuron_id].append(max_t)
    # each row encodes a column population, don't transpose it because it is easier to
    # iterate over rows (needed in Retina spike times initialisation)
    retina_spikes['left'] = np.asarray(retina_spikes['left'])
    retina_spikes['right'] = np.asarray(retina_spikes['right'])
    return retina_spikes


def save_spikes(output_dir, spikes, experiment_name='experiment'):
    """
    Save the spikes to a pickle file. 
    
    Args:
        output_dir: str, the output directory where the spikes file will be written
        spikes: a dict containing the spikes with pixel coordinates, disparity values and timestamps
        experiment_name: str, the name of the experiment used to encode the filename

    Returns:
        The full path to the pickled object
    """
    output_folder_path = output_dir
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    timestamp = str(dt.now().isoformat())
    filename = os.path.join(output_folder_path, experiment_name + "_tcd-out_" + timestamp + ".pickle")
    with open(filename, 'wb') as f:
        logger.debug("Saving {} spikes file.".format(filename))
        cPickle.dump(spikes, f)
    return filename
