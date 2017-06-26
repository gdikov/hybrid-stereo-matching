import logging
import os

import spynnaker.pyNN as pyNN

from framed.stereo_framebased import FramebasedStereoMatching
from spiking.optical_flow_correction import OpticalFlowPixelCorrection
from spiking.nn_modules.spike_source import create_retina
from spiking.stereo_snn import CooperativeNetwork
from stereovis.utils.helpers import latest_file_in_dir
from stereovis.utils.config import load_config
from stereovis.utils.frames_io import load_frames, save_frames, generate_frames_from_spikes
from stereovis.utils.spikes_io import load_spikes, save_spikes

logger = logging.getLogger(__file__)


class HybridStereoMatching(object):
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
        if self.config['general']['mode'] == 'online':
            raise NotImplementedError("Online operational mode is not supported yet.")

        self.effective_frame_resolution = (self.config['input']['resolution'][0] //
                                           self.config['input']['scale_down_factor'][0],
                                           self.config['input']['resolution'][1] //
                                           self.config['input']['scale_down_factor'][1])
        self.prior_buffer_interval = 1000 // (self.config['input']['frame_rate'] *
                                              self.config['input']['buffer_frame_ratio'])

        if self.config['simulation']['run_eventbased']:
            logger.info("Preparing events for the event-based stereo matching.")
            # setup timestep of simulation and minimum and maximum synaptic delays
            pyNN.setup(timestep=self.config['simulation']['timestep'],
                       min_delay=self.config['simulation']['timestep'],
                       max_delay=10 * self.config['simulation']['timestep'],
                       n_chips_required=self.config['simulation']['n_chips'],
                       threads=4)

            logger.info("Preparing spike sources for offline mode.")
            retina_spikes = load_spikes(input_file=self.config['input']['spikes_path'],
                                        resolution=self.config['input']['resolution'],
                                        crop_region=self.config['input']['crop'],
                                        simulation_time=self.config['simulation']['duration'],
                                        timestep_unit=self.config['input']['timestamp_unit'],
                                        dt_thresh=1,
                                        scale_down_factor=self.config['input']['scale_down_factor'],
                                        as_spike_source_array=True)
            retina_left = create_retina(retina_spikes['left'], label='retina_left')
            retina_right = create_retina(retina_spikes['right'], label='retina_right')
            spiking_inputs_resolution = (len(retina_left), len(retina_left[0]))
            spiking_inputs = {'left': retina_left, 'right': retina_right, 'resolution': spiking_inputs_resolution}

            logger.info("Setting up spiking neural network for event-based stereo matching.")
            if self.config['general']['eventbased_algorithm'] == 'tcd':
                self.eventbased_algorithm = CooperativeNetwork(spiking_inputs,
                                                               experiment_config=self.config['input'],
                                                               params=self.config['general']['network_params'],
                                                               operational_mode=self.config['general']['mode'])
            else:
                raise ValueError("Unsupported event-based algorithm. Currently only `tcd` (temporal coincidence "
                                 "detection) is supported.")
        else:
            logger.info("Skipping spiking stereo matching algorithm initialisation.")

        if self.config['simulation']['run_framebased']:
            logger.info("Preparing frames for the frame-based stereo matching.")
            frames_left, times = load_frames(input_path=os.path.join(self.config['input']['frames_path'], 'left'),
                                             resolution=self.config['input']['resolution'],
                                             crop_region=self.config['input']['crop'],
                                             scale_down_factor=self.config['input']['scale_down_factor'],
                                             simulation_time=self.config['simulation']['duration'],
                                             timestamp_unit=self.config['input']['timestamp_unit'])
            frames_right, _ = load_frames(input_path=os.path.join(self.config['input']['frames_path'], 'right'),
                                          resolution=self.config['input']['resolution'],
                                          crop_region=self.config['input']['crop'],
                                          scale_down_factor=self.config['input']['scale_down_factor'],
                                          simulation_time=self.config['simulation']['duration'],
                                          timestamp_unit=self.config['input']['timestamp_unit'])
            # save_frames(frames_left, os.path.join(self.config['general']['output_dir'], 'frames_left'))
            # save_frames(frames_right, os.path.join(self.config['general']['output_dir'], 'frames_right'))
            logger.info("Setting up MRF belief propagation network for frame-based stereo matching.")
            self.framebased_algorithm = \
                FramebasedStereoMatching(algorithm=self.config['general']['framebased_algorithm'],
                                         resolution=self.effective_frame_resolution,
                                         max_disparity=self.config['input']['max_disparity'],
                                         inputs={'left': frames_left, 'right': frames_right, 'ts': times})
            if self.config['general']['use_prior_adjustment']:
                logger.info("Setting up Velocity Field estimation for prior adjustment.")
                retina_spikes = load_spikes(input_file=self.config['input']['spikes_path'],
                                            resolution=self.config['input']['resolution'],
                                            crop_region=self.config['input']['crop'],
                                            simulation_time=self.config['simulation']['duration'],
                                            timestep_unit=self.config['input']['timestamp_unit'],
                                            dt_thresh=1,
                                            scale_down_factor=self.config['input']['scale_down_factor'],
                                            as_spike_source_array=False)
                self.prior_adjustment_algorithm = OpticalFlowPixelCorrection(resolution=self.effective_frame_resolution,
                                                                             reference_events=retina_spikes['left'],
                                                                             buffer_pivots=times,
                                                                             buffer_interval=self.prior_buffer_interval)
            else:
                logger.info("Skipping prior adjustment algorithm initialisation.")
        else:
            logger.info("Skipping frame-based stereo matching algorithm initialisation.")

    def run(self):
        """
        Run the spiking network and the Markov random field according to the experiment configuration.
        
        Returns:
            In-place method.
        """
        if self.config['simulation']['run_eventbased']:
            self.eventbased_algorithm.run(self.config['simulation']['duration'])
            prior_disparities = self.eventbased_algorithm.get_output()
            save_spikes(self.config['general']['output_dir'], prior_disparities,
                        experiment_name=self.config['general']['name'])
            effective_frame_resolution = self.effective_frame_resolution
        else:
            logger.info("Loading pre-computed spiking network output.")
            eventbased_output = latest_file_in_dir(self.config['general']['output_dir'], extension='pickle')
            prior_disparities = load_spikes(eventbased_output)
            effective_frame_resolution = prior_disparities['meta']['resolution']

        if self.config['simulation']['run_framebased']:
            pivots = self.framebased_algorithm.get_timestamps()
            prior_frames, timestamps = generate_frames_from_spikes(resolution=effective_frame_resolution,
                                                                   xs=prior_disparities['xs'],
                                                                   ys=prior_disparities['ys'],
                                                                   ts=prior_disparities['ts'],
                                                                   zs=prior_disparities['disps'],
                                                                   time_interval=self.prior_buffer_interval,
                                                                   pivots=pivots,
                                                                   non_pixel_value=-1)
            if self.config['general']['use_prior_adjustment']:
                prior_frames = self.prior_adjustment_algorithm.adjust(prior_frames)
            save_frames(prior_frames, os.path.join(self.config['general']['output_dir'], 'prior_frames'))
            prior_dict = {'priors': prior_frames, 'ts': timestamps}
            self.framebased_algorithm.run(prior_dict)
            depth_frames = self.framebased_algorithm.get_output()
            # save_frames(depth_frames, self.config['general']['output_dir'])
