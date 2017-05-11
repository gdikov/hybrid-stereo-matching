import urllib
import re
import contextlib
import os
import numpy as np


class SpikeParser:
    def __init__(self, crop_region=None, resolution=None, simulation_time=None, timestep_unit='us'):
        """
        Args:
            crop_region: a tuple of top left and bottom right pixel coordinates, which will be cropped from the 
             left and right retinas respectively. 
            resolution: a tuple of retina's x and y resolution (pixel count)
            simulation_time: a tuple with start and end time or an int with an end time
            timestep_unit: the unit in which the events are recorded
        """
        self.spikes_left = []
        self.spikes_right = []

        # define the crop region by a top left and bottom right corner coordinates
        if crop_region is not None:
            self.crop_region = crop_region
        else:
            dx, dy = resolution
            self.crop_region = (0, 0) + (dx-1, dy-1)
        self.effective_resolution = (self.crop_region[3] - self.crop_region[1],
                                     self.crop_region[2] - self.crop_region[0])

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
                # interpret 0 as right retina id
                if event[-1] == 0:
                    events['right'].append(event)
                else:
                    events['left'].append(event)
            return events

        if is_input_url:
            # connect to website and parse text data
            with contextlib.closing(urllib.urlopen(input_data)) as fp:
                raw_data = fp.read()
                raw_data = raw2num(raw_data)
        else:
            filename = os.path.basename(input_data)
            if filename.endswith('dat') or filename.endswith('txt'):
                with open(input_data, 'r') as fp:
                    raw_data = fp.read()
                    raw_data = raw2num(raw_data)
            elif filename.endswith('npz') or filename.endswith('npy'):
                # a dict with 'left' and 'right' keys containing a numpy arrays of N spikes
                # each of which is represented by a timestamp, x, y coordinate and a polarity value.
                raw_data = np.load(input_data)
            else:
                ValueError("Unknown file type. Provide a .txt, .dat, .npz file or url.")

        # needs to be sorted by timestamp
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
        # initialise the maximum time constant as the total simulation duration.
        # This is needed to set a value for pixels which don't spike at all, since the pyNN frontend requires so.
        # If they spike at the last possible time step, their firing will have no effect on the simulation.

        # convert time to ms
        if self.time_unit == 'us':
            events['left'][:, 0] //= 1000
            events['right'][: 0] //= 1000

        if not assume_sorted:
            events['left'] = events['left'][events['left'][:, 0].argsort()]
            events['right'] = events['right'][events['right'][:, 0].argsort()]

        # get the time region of interest
        if self.simulation_end_time is not None:
            events['left'] = events['left'][np.argmax(events['left'][:, 0] >= self.simulation_start_time):
                                            np.argmax(events['left'][:, 0] > self.simulation_end_time), :]
            events['right'] = events['right'][np.argmax(events['right'][:, 0] >= self.simulation_start_time):
                                              np.argmax(events['right'][:, 0] > self.simulation_end_time), :]

        # filter event bursts, i.e. spikes which occur faster than a dt_threshold
        dts_left = np.convolve(events['left'][:, 0], [1, -1], mode='valid')
        events['left'] = events['left'][np.concatenate([[True], dts_left > dt_threshold])]
        dts_right = np.convolve(events['right'][:, 0], [1, -1], mode='valid')
        events['right'] = events['right'][np.concatenate([[True], dts_right > dt_threshold])]

        # filter events which lie outside the crop_region
        events['left'] = events['left'][(self.crop_region[0] <= events['left'][:, 1] <= self.crop_region[2])
                                        & (self.crop_region[1] <= events['left'][:, 2] <= self.crop_region[3])]
        events['right'] = events['right'][(self.crop_region[0] <= events['right'][:, 1] <= self.crop_region[2])
                                          & (self.crop_region[1] <= events['right'][:, 2] <= self.crop_region[3])]
        return events


def load_spikes(input_file, crop_region=None, resolution=None, simulation_time=None, timestep_unit='us', dt_thresh=0):
    """
    Load the spikes form a file or a url into a list of populations each with lists of neuron spiking times. 
     
    Args:
        input_file: path to input data file or a url
        crop_region: the region of interest in pixels (top left and bottom right coordinates in x, y order)
        resolution: the standard retina resolution
        simulation_time: simulation start and end time or simulation end time only
        timestep_unit: the units in which the timestamps are encoded. Can be 'us' or 'ms'.
        dt_thresh: shortest amount of time in which same pixels are not allowed to spike. 

    Returns:
        A dict with 'left' and 'right' retina spiking times.
    """
    parser = SpikeParser(crop_region, resolution, simulation_time, timestep_unit)
    raw_data = parser.parse(input_file)
    if crop_region is not None or simulation_time is not None or dt_thresh > 0:
        filtered_data = parser.sanitise_events(raw_data, dt_thresh, assume_sorted=True)
    # create lists of (populations) lists of (neuron's spiking times) lists and fill in the time values
    max_t = np.max([filtered_data['left'][:, 0], filtered_data['right'][:, 0]]) + 1
    n_cols, n_rows = parser.effective_resolution
    retina_spikes = {'left': [[[] for _ in xrange(n_rows)] for _ in xrange(n_cols)],
                     'right': [[[] for _ in xrange(n_rows)] for _ in xrange(n_cols)]}
    for t, neuron_id, population_id, _ in filtered_data['left']:
        retina_spikes['left'][population_id][neuron_id].append(t)
    for t, neuron_id, population_id, _ in filtered_data['left']:
        retina_spikes['right'][population_id][neuron_id].append(t)
    # append a fictitious spiking time which is never reached.
    # It is necessary for the SpikeSourceArray needs an initialisation for each neuron's spiking
    for population_id in xrange(n_cols):
        for neuron_id in xrange(n_rows):
            retina_spikes['left'][population_id][neuron_id].append(max_t)
            retina_spikes['right'][population_id][neuron_id].append(max_t)
    return retina_spikes
