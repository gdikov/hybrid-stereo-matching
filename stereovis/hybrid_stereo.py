import os
import spynnaker.pyNN as pyNN
from spiking.stereo_snn import BasicCooperativeNetwork, HierarchicalCooperativeNetwork
from spiking.nn_modules.spike_source import create_retina
from framed.stereo_framebased import FramebasedStereoMatching

from utils.spikes_io import load_spikes, save_spikes
from utils.frames_io import load_frames, save_frames, generate_frames_from_spikes
from utils.config import load_config
from utils.helpers import latest_file_in_dir

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
        if self.config['simulation']['run_eventbased']:
            logger.info("Preparing events for the event-based stereo matching.")
            # setup timestep of simulation and minimum and maximum synaptic delays
            pyNN.setup(timestep=self.config['simulation']['timestep'],
                       min_delay=self.config['simulation']['timestep'],
                       max_delay=10 * self.config['simulation']['timestep'],
                       n_chips_required=self.config['simulation']['n_chips'],
                       threads=4)
        else:
            logger.info("Skipping spiking stereo matching algorithm initialisation.")
        if self.config['general']['mode'] == 'offline':
            if self.config['simulation']['run_eventbased']:
                logger.info("Preparing spike sources for offline mode.")
                spikes = load_spikes(input_file=self.config['input']['spikes_path'],
                                     resolution=self.config['input']['resolution'],
                                     crop_region=self.config['input']['crop'],
                                     simulation_time=self.config['simulation']['duration'],
                                     timestep_unit=self.config['input']['timestamp_unit'],
                                     dt_thresh=1,
                                     scale_down_factor=self.config['input']['scale_down_factor'],
                                     as_spike_source_array=True)
                retina_left = create_retina(spikes['left'], label='retina_left')
                retina_right = create_retina(spikes['right'], label='retina_right')
                spiking_inputs_resolution = (len(retina_left), len(retina_left[0]))
                spiking_inputs = {'left': retina_left, 'right': retina_right, 'resolution': spiking_inputs_resolution}
            if self.config['simulation']['run_framebased']:
                if self.config['general']['framebased_algorithm'] != 'none':
                    logger.info("Preparing frames for the frame-based stereo matching.")
                    frames_left, times = load_frames(input_path=os.path.join(self.config['input']['frames_path'], 'left'),
                                                     resolution=self.config['input']['resolution'],
                                                     crop_region=self.config['input']['crop'],
                                                     simulation_time=self.config['simulation']['duration'],
                                                     timestamp_unit=self.config['input']['timestamp_unit'])
                    frames_right, _ = load_frames(input_path=os.path.join(self.config['input']['frames_path'], 'right'),
                                                  resolution=self.config['input']['resolution'],
                                                  crop_region=self.config['input']['crop'],
                                                  simulation_time=self.config['simulation']['duration'],
                                                  timestamp_unit=self.config['input']['timestamp_unit'])
                    logger.info("Setting up MRF belief propagation network for frame-based stereo matching.")
                    self.framebased_algorithm = \
                        FramebasedStereoMatching(algorithm=self.config['general']['framebased_algorithm'],
                                                 resolution=self.config['input']['resolution'],
                                                 max_disparity=self.config['input']['max_disparity'],
                                                 frames={'left': frames_left, 'right': frames_right, 'ts': times})
                else:
                    raise ValueError("The configuration parameters make no sense. "
                                     "When running a frame-based stereo matching, provide also the algorithm type.")
            else:
                logger.warning("Skipping frame-based stereo matching algorithm initialisation.")
        else:
            raise NotImplementedError("Online mode of operation is not upproted yet.")

        if self.config['simulation']['run_eventbased']:
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

    def run(self):
        """
        Run the spiking network and the Markov random field according to the experiment configuration.
        
        Returns:
            In-place method.
        """
        if self.config['general']['mode'] == 'offline':
            if self.config['simulation']['run_eventbased']:
                self.eventbased_algorithm.run(self.config['simulation']['duration'])
                prior_disparities = self.eventbased_algorithm.get_output()
                save_spikes(self.config['general'], prior_disparities)
            if self.config['simulation']['run_framebased']:
                if not self.config['simulation']['run_eventbased']:
                    logger.info("Loading pre-computed spiking network output.")
                    eventbased_output = latest_file_in_dir(self.config['general']['output_dir'], extension='pickle')
                    prior_disparities = load_spikes(eventbased_output)
                prior_buffer_interval = 1000 // self.config['input']['frame_rate'] // 4
                prior_frames, timestamps = generate_frames_from_spikes(resolution=self.config['input']['resolution'],
                                                                       xs=prior_disparities['xs'],
                                                                       ys=prior_disparities['ys'],
                                                                       ts=prior_disparities['ts'],
                                                                       zs=prior_disparities['disps'],
                                                                       time_interval=prior_buffer_interval,
                                                                       pivots=self.framebased_algorithm.get_timestamps(),
                                                                       non_pixel_value='nan')
                save_frames(prior_frames, os.path.join(self.config['general']['output_dir'], 'prior_frames'))
                prior_dict = {'priors': prior_frames, 'ts': timestamps}
                self.framebased_algorithm.run(prior_dict)
                depth_frames = self.framebased_algorithm.get_output()
                save_frames(depth_frames, self.config['general']['output_dir'])
        else:
            raise NotImplementedError

