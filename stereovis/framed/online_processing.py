import ctypes
import time

import numpy as np
import multiprocessing as mp

from multiprocessing import sharedctypes


class OnlineMatching(object):
    def __init__(self):
        # self.process_communication_receive, self.process_communication_send = mp.Pipe(duplex=False)
        self.np_buffer = None
        self.times_placeholder = None
        self.matching_deamon = None
        pass

    def init_shared_buffer(self, size):
        # prefer shared memory to server manager only for the speed advantage
        buffer_ = mp.sharedctypes.RawArray(ctypes.c_int32, np.zeros(np.prod(size), dtype=np.int32))
        np_buffer = np.frombuffer(buffer_, dtype=np.int32)
        self.np_buffer = np_buffer.reshape(size)
        self.times_placeholder = mp.sharedctypes.RawValue(ctypes.c_int32, 0)
        return self.np_buffer, self.times_placeholder

    def matching_loop(self):
        old_time = self.times_placeholder.value
        while True:
            if old_time < self.times_placeholder.value:
                print(self.times_placeholder.value)
                old_time = self.times_placeholder.value

    def run(self):
        self.matching_deamon = mp.Process(target=self.matching_loop)
        self.matching_deamon.start()

    def end(self):
        self.matching_deamon.terminate()

    def join(self):
        self.matching_deamon.join()
