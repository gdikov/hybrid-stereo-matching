from __future__ import division

import logging
import os

try:
    import spynnaker7.pyNN as pyNN
except ImportError:
    import spynnaker.pyNN as pyNN

from numpy import save as save_array

from framed.stereo_framebased import FramebasedStereoMatching
from framed.online_processing import OnlineMatching
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
            logger.warning("Currently the online operational mode is supported only for live output.")

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
                self.network_meta = self.eventbased_algorithm.get_meta_info()
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
        if self.config['general']['mode'] == 'offline':
            self._run_offline()
        else:
            try:
                self._run_online()
            except RuntimeError:
                logger.warning("Unexpected error during online mode. Restarting experiment in offline mode.")
                self._run_offline()

    def _run_offline(self):
        """
        Run the SNN network if desired and perform MRF stereo-matching afterwards.

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
            prior_nondata_value = -1
            prior_frames, timestamps = generate_frames_from_spikes(resolution=effective_frame_resolution,
                                                                   xs=prior_disparities['xs'],
                                                                   ys=prior_disparities['ys'],
                                                                   ts=prior_disparities['ts'],
                                                                   zs=prior_disparities['disps'],
                                                                   time_interval=self.prior_buffer_interval,
                                                                   pivots=pivots,
                                                                   non_pixel_value=prior_nondata_value)
            if self.config['general']['use_prior_adjustment']:
                prior_frames = self.prior_adjustment_algorithm.adjust(prior_frames,
                                                                      prior_nondata_value=prior_nondata_value,
                                                                      time_arrow=-1)
            save_frames(prior_frames, os.path.join(self.config['general']['output_dir'], 'prior_frames'))
            prior_dict = {'priors': prior_frames, 'ts': timestamps}
            self.framebased_algorithm.run(prior_dict)
            depth_frames = self.framebased_algorithm.get_output()
            save_frames(depth_frames, self.config['general']['output_dir'])

    def _run_online(self):
        """
       Run the spiking network and the Markov random field at the same time, using the live output stream from the SNN.

       Returns:
           In-place method.

       Notes:
           The online mode is limited to using only live SNN output spikes.
       """
        try:
            from spynnaker_external_devices_plugin.pyNN.connections.spynnaker_live_spikes_connection import \
                SpynnakerLiveSpikesConnection
        except ImportError:
            logger.warning("Spynnaker external modules are not installed. Exiting online mode.")
            raise RuntimeError

        def _gather_spikes(label, timestamp, neuron_ids):
            """
            Callback for the event processing.

            Args:
                label: str, the label of the population form which the event originates
                timestamp: int, timestamp in milliseconds for the event
                neuron_ids: list, one or mode neuron ids (int) which produced the spikes

            Returns:
                In-place method

            Notes:
                Keep this callback short and quick!
            """
            population_id = int(label.split('_')[1])
            for neuron_id in neuron_ids:
                disp = self.network_meta['id2disparity'][population_id]
                col, row = self.network_meta['id2pixel'][population_id], neuron_id
                # store the last spikes and its timestamp in the shared memory event and timestamp buffer
                # notice that the event buffer is a flattened 2d array.
                self.spikes_buffer[row * self.effective_frame_resolution[0] + col] = disp
                self.spikes_ts.value = timestamp

        def _init_simulation(label, event_connection):
            """
            Callback for the beginning of the simulation. Set the shared boolean flag.

            Args:
                label: str, unused argument
                event_connection: SpyNNakerLiveEventConnection, unused argument

            Returns:
                In-place method.
            """
            self.simulation_started.value = True

        live_connection = SpynnakerLiveSpikesConnection(receive_labels=self.network_meta['collector_labels'],
                                                        local_port=19996,
                                                        send_labels=None)
        for label in self.network_meta['collector_labels']:
            live_connection.add_receive_callback(label, _gather_spikes)
            try:
                live_connection.add_start_resume_callback(label, _init_simulation)
            except AttributeError:
                # using an older sPyNNaker release
                live_connection.add_start_callback(label, _init_simulation)

        matcher = OnlineMatching(algorithm=self.framebased_algorithm,
                                 simulation_time_step=self.config['simulation']['timestep'],
                                 snn_slow_down_factor=self.config['simulation']['time_scale_factor'],
                                 frame_length=1000 // self.config['input']['frame_rate'])
        self.spikes_buffer, self.spikes_ts, self.simulation_started = matcher.init_shared_buffer(
            buffer_shape=self.effective_frame_resolution[::-1])
        with matcher.run():
            try:
                self.eventbased_algorithm.run(self.config['simulation']['duration'])
            except Exception as e:
                logger.error("An error occured during simulation or compilation: '{}'".format(e))
        prior_posterior = zip(*matcher.get_output())
        if not prior_posterior:
            logger.warning("No output from the SNN has been detected.")
            return
        else:
            save_array(os.path.join(self.config['general']['output_dir'], 'priors.npy'), prior_posterior[0])
            save_array(os.path.join(self.config['general']['output_dir'], 'posteriors.npy'), prior_posterior[1])
            save_frames(prior_posterior[0], os.path.join(self.config['general']['output_dir'], 'priors'))
            save_frames(prior_posterior[1], os.path.join(self.config['general']['output_dir'], 'posteriors'))
