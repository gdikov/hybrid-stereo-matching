import spynnaker.pyNN as pyNN
from nn_modules.temporal_coincidence import TemporalCoincidenceDetectionNetwork
from nn_modules.spatial_constraints import EventFrameFusionNetwork
from nn_modules.spike_source import Retina
from utils.spikes_io import SpikeParser


class CooperativeNetwork:
    def __init__(self, spike_sources):
        # setup timestep of simulation and minimum and maximum synaptic delays
        simulation_time_step = 0.2
        pyNN.setup(timestep=simulation_time_step,
                   min_delay=simulation_time_step,
                   max_delay=10 * simulation_time_step,
                   n_chips_required=6,
                   threads=4)
        self.retina_inputs = {'left': spike_sources['left'], 'right': spike_sources['right']}

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
    def __init__(self, spiking_inputs, operational_mode):
        CooperativeNetwork.__init__(self, spiking_inputs)
        self.network = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs, mode=operational_mode)

    def get_output(self):
        return self.network.get_output()
