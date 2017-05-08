from spiking.stereo_snn import BasicCooperativeNetwork
from spiking.nn_modules.spike_source import Retina
from framed.stereo_mrf import StereoMRF
from framed.frame_manager import FrameManager

import logging

logger = logging.getLogger(__file__)


# FIXME: add online mode of working too!
class HybridStereoMatching:
    def __init__(self, mode='offline'):
        self.mode = mode
        if mode == 'offline':
            spiking_inputs = {'left': Retina(spike_times=None), 'right': Retina(spike_times=None)}
            self.snn = BasicCooperativeNetwork(spiking_inputs, mode)
            self.mrf = StereoMRF()
            self.frames = FrameManager()
        elif mode == 'online':
            logger.error("Online mode of operation is not supported yet. Rerun in `offline` mode.")
            raise ValueError
        else:
            raise ValueError("Mode can be only `online` or `offline`.")

    def run(self):
        if self.mode == 'offline':
            # while True: get frame -> run MRF -> get result and push to SNN -> get result and init nex MRF cycle.

            for left_frame, right_frame, timestamp in self.frames.iter_frames():
                snn_output = self.snn.get_output(time=timestamp)
                self.mrf.loop_belief(image_left=left_frame, image_right=right_frame,
                                     prior=snn_output, n_iter=2, reinit_messages=False)
