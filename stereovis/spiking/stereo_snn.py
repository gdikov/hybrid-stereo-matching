import spynnaker.pyNN as pyNN
from nn_modules.temporal_coincidence import TemporalCoincidenceDetectionNetwork
from nn_modules.spatial_constraints import EventFrameFusionNetwork
from nn_modules.spike_source import Retina
from utils.spikes_io import SpikeParser


class CooperativeNetwork:
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



class HierarchicalCooperativeNetwork(CooperativeNetwork):
    def __init__(self):
        CooperativeNetwork.__init__(self)
        self.temporal_net = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs)
        # TODO: init the fusion network


class BasicCooperativeNetwork(CooperativeNetwork):
    def __init__(self, spiking_inputs, params=None, experiment_config=None, operational_mode='offline'):
        CooperativeNetwork.__init__(self, spiking_inputs)
        self.network = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs,
                                                           network_params=params,
                                                           experiment_params=experiment_config,
                                                           mode=operational_mode)

    def get_output(self):
        return self.network.get_output()
