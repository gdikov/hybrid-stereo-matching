import os
import logging

import spynnaker.pyNN as pyNN

from utils.config import load_config

logger = logging.getLogger(__file__)


class Retina(object):
    def __init__(self, spike_times=None, params=None, label='retina'):

        path_to_params = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config")
        if params is None:
            self.params = load_config(os.path.join(path_to_params, 'default_params.yaml'))
        else:
            self.params = load_config(os.path.join(path_to_params, params + '_params.yaml'))

        if len(spike_times) >= self.params['retina']['n_cols'] or \
           len(spike_times[0]) >= self.params['retina']['n_cols']:
            logger.error("Dimensionality of the retina spiking array is not matching the parameter configuration.")
            raise ValueError

        self.pixel_populations = []
        self.labels = []

        if spike_times is not None:
            for x in range(0, self.params['retina']['n_cols'] - self.params['topology']['min_disparity']):
                col_of_pixels = pyNN.Population(size=self.params['retina']['n_rows'],
                                                cellclass=pyNN.SpikeSourceArray,
                                                cellparams={'spike_times': spike_times[x]},
                                                label="{0}_{1}".format(label, x), structure=pyNN.Line())

                self.pixel_populations.append(col_of_pixels)
        else:
            logger.error("Live input streaming is not supported yet. Provide a spike_times array instead.")
