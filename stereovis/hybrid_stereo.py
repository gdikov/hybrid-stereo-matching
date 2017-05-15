import spynnaker.pyNN as pyNN
from spiking.stereo_snn import BasicCooperativeNetwork, HierarchicalCooperativeNetwork
from spiking.nn_modules.spike_source import create_retina
from framed.stereo_mrf import StereoMRF

from utils.spikes_io import load_spikes, save_spikes
from utils.frames_io import load_frames
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
        self._setup()

    def _setup(self):
        """
        Setup the spiking neural network and the frame-based stereo matching from config file.

        Returns:
            In-place method
        """
        logger.info("Preparing events for the event-based stereo matching.")
        # setup timestep of simulation and minimum and maximum synaptic delays
        pyNN.setup(timestep=self.config['simulation']['timestep'],
                   min_delay=self.config['simulation']['timestep'],
                   max_delay=10 * self.config['simulation']['timestep'],
                   n_chips_required=self.config['simulation']['n_chips'],
                   threads=4)
        if self.config['general']['mode'] == 'offline':
            logger.info("Preparing spike sources for offline mode.")
            spikes = load_spikes(input_file=self.config['input']['spikes_path'],
                                 resolution=self.config['input']['resolution'],
                                 crop_region=self.config['input']['crop'],
                                 simulation_time=self.config['simulation']['duration'],
                                 timestep_unit=self.config['input']['timestamp_unit'], dt_thresh=1)
            retina_left = create_retina(spikes['left'], label='retina_left')
            retina_right = create_retina(spikes['right'], label='retina_right')
            spiking_inputs = {'left': retina_left, 'right': retina_right}
            if self.config['general']['framebased_algorithm'] != 'none':
                logger.info("Preparing frames for the frame-based stereo matching.")
                # self.frames = FrameManager()
            else:
                logger.warning("Skipping frame-based stereo matching algorithm initialisation. "
                               "The Hybrid Stereo Matching framework is reduced to purely event-based one.")
        else:
            raise NotImplementedError("Online mode of operation is not upproted yet.")

        logger.info("Setting up spiking neural network for event-based stereo matching.")
        if self.config['general']['eventbased_algorithm'] == 'tcd':
            self.eventbased_algorithm = BasicCooperativeNetwork(spiking_inputs,
                                                                experiment_config=self.config['input'],
                                                                params=self.config['general']['network_params'],
                                                                operational_mode=self.config['general']['mode'])
        elif self.config['general']['eventbased_algorithm'] == 'hn':
            self.eventbased_algorithm = HierarchicalCooperativeNetwork(spiking_inputs,
                                                                       experiment_config=self.config['input'],
                                                                       params=self.config['general']['network_params'],
                                                                       operational_mode=self.config['general']['mode'])
        else:
            raise ValueError("Unsupported event-based algorithm. Can be `tcd` for the temporal coincidence "
                             "detection only or `hn` for a hierarchical network architecture.")

        # logger.info("Setting up MRF belief propagation network for frame-based stereo matching.")
        # self.framebased_algorithm = StereoMRF()


    def run(self):
        """
        Run the spiking network and the Markov random field.
        
        Returns:

        """
        logger.info("Starting the spiking neural network.")
        self.eventbased_algorithm.run(self.config['simulation']['duration'])
        prior_disparities = self.eventbased_algorithm.get_output()
        save_spikes(self.config['general'], prior_disparities)
        self.framebased_algorithm.run(prior_disparities)

