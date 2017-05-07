import os, logging
import spynnaker.pyNN as pyNN

from utils.params import load_params

logger = logging.getLogger(__file__)


class EventFrameFusionNetwork:
    def __init__(self, input_sources=None, params=None):
        path_to_params = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config")
        if params is None:
            self.params = load_params(os.path.join(path_to_params, 'default_params.yaml'))
        else:
            self.params = load_params(os.path.join(path_to_params, 'default' + '_params.yaml'))

