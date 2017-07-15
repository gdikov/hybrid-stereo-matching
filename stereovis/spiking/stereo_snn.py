try:
    import spynnaker7.pyNN as pyNN
except ImportError:
    import spynnaker.pyNN as pyNN
from nn_modules.temporal_coincidence import TemporalCoincidenceDetectionNetwork
# from nn_modules.spatial_constraints import EventFrameFusionNetwork

import logging

logger = logging.getLogger(__file__)


class SpikingNeuralNetwork:
    def __init__(self, spike_sources):
        self.retina_inputs = spike_sources

    def get_output(self):
        """
        Return all the spikes in an interpretable form. If no spike -> disparity mapping is desired, then use the
        `get_raw_output()` method instead.
         
        Returns:
            A list of spiking times, x, y coordinate and disparity value.
        
        Notes:
            In the output list the x and y coordinates are referring to the left retina pixel space.      
        """
        return None

    @staticmethod
    def run(duration=None):
        """
        Run the simulation for `duration` milliseconds
        
        Args:
            duration: the number of milliseconds for the duration of the experiment (simulation). 
             If None then it will run indefinitely long. 
        
        """
        pyNN.run(run_time=duration)

    @staticmethod
    def end():
        """
        Clean-up after simulation ends.
        
        """
        pyNN.end()


class CooperativeNetwork(SpikingNeuralNetwork):
    def __init__(self, spiking_inputs, params=None, experiment_config=None, operational_mode='offline'):
        SpikingNeuralNetwork.__init__(self, spiking_inputs)
        logger.debug("Creating Basic Cooperative Network instance, consisting "
                     "in a Temporal Coincidence Detection module only.")
        self.network = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs,
                                                           network_params=params,
                                                           experiment_params=experiment_config,
                                                           mode=operational_mode)

    def get_output(self):
        return self.network.get_output()

    def get_meta_info(self):
        meta = {'collector_labels': self.network.get_labels()}
        for k, v in self.network.get_mappings().items():
            meta[k] = v
        return meta
