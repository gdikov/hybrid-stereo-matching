from __future__ import division

import ctypes
import logging
import contextlib
import time

import numpy as np
import multiprocessing as mp

from multiprocessing import sharedctypes

logger = logging.getLogger(__file__)
np.set_printoptions(threshold=np.nan)


class OnlineMatching(object):
    def __init__(self, algorithm, snn_slow_down_factor=1.0, frame_length=50):
        self.framebased_module = algorithm
        self.slow_down_factor = snn_slow_down_factor
        # constant time buffer around the frame timestamp which ensures
        # that triggering of the MRF BP loop will not start too late
        self.time_epsilon = 1
        self.nominal_frame_length = frame_length
        self.buffer_ = None
        self.times_placeholder = None
        self.matching_deamon = None
        self.buffer_shape = None

    def init_shared_buffer(self, buffer_shape):
        self.buffer_shape = buffer_shape
        # prefer shared memory to server manager only for the speed advantage
        self.buffer_ = mp.sharedctypes.RawArray(ctypes.c_int32, np.ones(np.prod(buffer_shape), dtype=np.int32) * -1)
        self.times_placeholder = mp.sharedctypes.RawValue(ctypes.c_int32, 0)
        self.simulation_started = mp.sharedctypes.RawValue(ctypes.c_bool, False)
        return self.buffer_, self.times_placeholder, self.simulation_started

    def _reset_buffer(self):
        for i in range(np.prod(self.buffer_shape)):
            self.buffer_[i] = -1

    def frame_generator(self):
        for left, right, timestamp in zip(self.framebased_module.frames_left,
                                          self.framebased_module.frames_right,
                                          self.framebased_module.frames_timestamps):
            yield left, right, timestamp

    def matching_loop(self):
        epoch_pc = time.time() * 1000
        for left, right, timestamp in self.frame_generator():
            while not self.simulation_started:
                # loop trap for synchronization purposes
                epoch_pc = time.time() * 1000
            tick_pc = (time.time() * 1000 - epoch_pc) / self.slow_down_factor
            tick_snn = self.times_placeholder.value / self.slow_down_factor
            while tick_snn < timestamp - self.time_epsilon and tick_pc < timestamp - self.time_epsilon:
                tick_pc = (time.time() * 1000 - epoch_pc) / self.slow_down_factor
                tick_snn = self.times_placeholder.value / self.slow_down_factor
            mrf_start = time.time()
            prior = np.frombuffer(self.buffer_, dtype=np.int32).reshape(self.buffer_shape)
            self.framebased_module.run_one_frame(left, right, prior, n_iter=10)
            self._reset_buffer()
            mrf_duration = (time.time() - mrf_start) * 1000
            logger.info("Images at the {}-th ms have been processed in {} ms out of {} ms available time"
                        .format(timestamp, mrf_duration, self.nominal_frame_length * self.slow_down_factor))

    @contextlib.contextmanager
    def run(self):
        self._reset_buffer()
        self.matching_deamon = mp.Process(target=self.matching_loop)
        self.matching_deamon.start()
        logger.info("Matching thread has been started.")
        yield
        self.end()

    def end(self):
        self.matching_deamon.terminate()
        logger.info("Matching thread has been stopped.")

    def join(self):
        logger.info("Matching thread has been joined.")
        self.matching_deamon.join()
