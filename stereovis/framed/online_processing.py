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
    def __init__(self, algorithm, snn_slow_down_factor=1.0, frame_length=50,
                 use_adaptive_iter=False, max_iter=10, min_iter=1):
        self.framebased_module = algorithm
        self.slow_down_factor = snn_slow_down_factor
        self.use_adaptive_iter = use_adaptive_iter
        self.max_iter = max_iter
        self.min_iter = min_iter
        # constant time buffer around the frame timestamp which ensures
        # that triggering of the MRF BP loop will not start too late
        self.time_epsilon = 1
        self.nominal_frame_length = frame_length
        self.buffer_ = None
        self.times_placeholder = None
        self.matching_deamon = None
        self.buffer_shape = None
        self.depth_frames = None
        self.prior_posterior = mp.Manager().list()

    def init_shared_buffer(self, buffer_shape):
        self.buffer_shape = buffer_shape
        # prefer shared memory to server manager only for the speed advantage
        self.buffer_ = mp.Array(ctypes.c_int32, np.ones(np.prod(buffer_shape), dtype=np.int32) * -1, lock=True)
        self.times_placeholder = mp.sharedctypes.RawValue(ctypes.c_int32, 0)
        self.simulation_started = mp.sharedctypes.RawValue(ctypes.c_bool, False)
        return self.buffer_, self.times_placeholder, self.simulation_started

    def _reset_buffer(self):
        for i in range(np.prod(self.buffer_shape)):
            self.buffer_[i] = -1

    def get_output(self):
        return self.prior_posterior

    def frame_generator(self):
        for left, right, timestamp in zip(self.framebased_module.frames_left,
                                          self.framebased_module.frames_right,
                                          self.framebased_module.frames_timestamps):
            yield left, right, timestamp

    def matching_loop(self):
        epoch_pc = time.time() * 1000
        mrf_duration = self.nominal_frame_length * self.slow_down_factor  # large int
        n_iter = 0
        for left, right, timestamp in self.frame_generator():
            while not self.simulation_started:
                # loop trap for synchronization purposes
                epoch_pc = time.time() * 1000
            # NOTE: there is ca. 1ms time difference between the SpiNNaker and the PC clock.
            # This can be compensated if necessary (e.g. using periodic synchronisation)
            tick_pc = (time.time() * 1000 - epoch_pc) / self.slow_down_factor
            tick_snn = self.times_placeholder.value / self.slow_down_factor
            while tick_snn < timestamp - self.time_epsilon and tick_pc < timestamp - self.time_epsilon:
                tick_pc = (time.time() * 1000 - epoch_pc) / self.slow_down_factor
                tick_snn = self.times_placeholder.value / self.slow_down_factor
            mrf_start = time.time()
            prior = np.frombuffer(self.buffer_.get_obj(), dtype=np.int32).reshape(self.buffer_shape)
            if self.use_adaptive_iter:
                t = mrf_duration / (self.nominal_frame_length * self.slow_down_factor)
                n_iter = int(n_iter + 0.2 * n_iter + 1 if t < 0.8 else n_iter - 0.5 * n_iter)
                n_iter = min(self.max_iter, max(self.min_iter, n_iter))
            else:
                n_iter = self.max_iter
            depth_map = self.framebased_module.run_one_frame(left, right, prior, n_iter=n_iter)
            mrf_duration = (time.time() - mrf_start) * 1000
            logger.info("Images at {} ms have been processed in {} iterations consuming {}/{} ms"
                        .format(timestamp, n_iter, mrf_duration, self.nominal_frame_length * self.slow_down_factor))
            self.prior_posterior.append((prior, depth_map))
            self._reset_buffer()
            if np.abs(tick_pc - tick_snn) > 1:
                logger.warning("Time discrepancy between SpiNNaker and PC "
                               "is greater than 1ms ({}ms)".format(tick_pc - tick_snn))
        logger.info("Images exhausted. {} prior-posterior pairs have been computed".format(len(self.prior_posterior)))

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
