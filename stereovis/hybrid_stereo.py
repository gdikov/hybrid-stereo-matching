import spynnaker.pyNN as pyNN
from spiking.stereo_snn import BasicCooperativeNetwork
from spiking.nn_modules.spike_source import create_retina
from framed.stereo_mrf import StereoMRF
from framed.frame_manager import FrameManager

from utils.spikes_io import load_spikes
from utils.config import load_config

import logging

logger = logging.getLogger(__file__)


# TODO: add online mode of working too!
class HybridStereoMatching:
    def __init__(self, experiment_config):
        """
        Args:
            experiment_config: a yaml file with the experiment configuration 
        """
        self.config = load_config(experiment_config)
        self.setup()

    def setup(self):
        """
        Setup the spiking neural network and the Markov random field from config file.

        Returns:
            In-place method
        """
        # setup timestep of simulation and minimum and maximum synaptic delays
        simulation_time_step = 0.2
        pyNN.setup(timestep=simulation_time_step,
                   min_delay=simulation_time_step,
                   max_delay=10 * simulation_time_step,
                   n_chips_required=6,
                   threads=4)
        spikes = load_spikes(input_file=self.config['input']['path'],
                             resolution=self.config['input']['resolution'],
                             crop_region=self.config['input']['crop'],
                             simulation_time=self.config['simulation']['duration'], timestep_unit='us', dt_thresh=1)
        retina_left = create_retina(spikes['left'], label='retina_left')
        retina_right = create_retina(spikes['right'], label='retina_right')
        spiking_inputs = {'left': retina_left, 'right': retina_right}
        self.snn = BasicCooperativeNetwork(spiking_inputs,
                                           experiment_config=self.config['input'],
                                           operational_mode='offline')
        # self.mrf = StereoMRF()
        # self.frames = FrameManager()

    def run(self):
        """
        Run the spiking network and the Markov random field.
        
        Returns:

        """
        self.snn.run(self.config['simulation']['duration'])
