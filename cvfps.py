import cv2
import numpy as np

def circularCounter(max):
    """helper function that creates an eternal counter till a max value"""
    x = 0
    while True:
        if x == max:
            x = 0
        x += 1
        yield x


class ringBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def push(self, x):
        "adds single element x to ring buffer"
        self.index = (self.index + 1) % self.data.size
        self.data[self.index] = x

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) %self.data.size
        return self.data[idx]


class cvTimer(object):
    def __init__(self, length=100, target_fps=30):
        self.last_tick = cv2.getTickCount()
        self.fpsP = cv2.getTickCount()
        self.fpsPLen = length
        # self.l_fps_history = [ target_fps for x in range(self.fpsPLen)]
        self.ringBuf = ringBuffer(length)

    def reset(self):
        self.last_tick = cv2.getTickCount()

    def get_tick_now(self):
        return cv2.getTickCount()

    @property
    def fps(self):
        self.fpsP = cv2.getTickFrequency() / float(self.get_tick_now() - self.last_tick)
        self.ringBuf.push(self.fpsP)
        self.reset()
        return self.fpsP

    @property
    def avg_fps(self):
        return sum(self.ringBuf.data) / float(self.fpsPLen)
