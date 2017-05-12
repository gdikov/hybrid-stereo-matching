from spiking.stereo_snn import BasicCooperativeNetwork
from spiking.nn_modules.spike_source import Retina
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

        if self.config['mode'] == 'offline':
            spikes = load_spikes(input_file=self.config['input']['path'],
                                 resolution=self.config['input']['resolution'],
                                 crop_region=self.config['input']['crop'],
                                 simulation_time=self.config['simulation']['duration'], timestep_unit='us', dt_thresh=1)
            spiking_inputs = {'left': Retina(spike_times=spikes['left']), 'right': Retina(spike_times=spikes['right'])}
        else:
            logger.error("Online mode of operation is not supported yet. Rerun in `offline` mode.")
            raise NotImplementedError
        self.snn = BasicCooperativeNetwork(spiking_inputs, self.config['mode'])
        # self.mrf = StereoMRF()
        # self.frames = FrameManager()

    def run(self):
        """
        Run the spiking network and the Markov random field.
        
        Returns:

        """
        if self.config['mode'] == 'offline':
            self.snn.run(self.config['simulation']['duration'])
