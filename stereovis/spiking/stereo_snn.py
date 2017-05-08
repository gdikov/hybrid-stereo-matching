from nn_modules.temporal_coincidence import TemporalCoincidenceDetectionNetwork
from nn_modules.spatial_constraints import EventFrameFusionNetwork


class CooperativeNetwork():
    def __init__(self, spiking_inputs=None, mode='offline'):
        self.retina_inputs = {'left': None, 'right': None}


class HierarchicalCooperativeNetwork(CooperativeNetwork):
    def __init__(self):
        CooperativeNetwork.__init__(self)
        self.temporal_net = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs)


class BasicCooperativeNetwork(CooperativeNetwork):
    def __init__(self, spiking_inputs, operational_mode):
        CooperativeNetwork.__init__(self, spiking_inputs, operational_mode)
        self.network = TemporalCoincidenceDetectionNetwork(input_sources=self.retina_inputs, mode=operational_mode)

