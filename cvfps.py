import cv2

def circular_counter(max):
    """helper function that creates an eternal counter till a max value"""
    x = 0
    while True:
        if x == max:
            x = 0
        x += 1
        yield x

class cvTimer(object):
    def __init__(self, length=100, target_fps=30):
        self.last_tick = cv2.getTickCount()
        self.fpsP = cv2.getTickCount()
        self.fpsPLen = length
        self.l_fps_history = [ target_fps for x in range(self.fpsPLen)]
        self.fpsPCounter = circular_counter(self.fpsPLen)

    def reset(self):
        self.last_tick = cv2.getTickCount()

    def get_tick_now(self):
        return cv2.getTickCount()

    @property
    def fps(self):
        self.fpsP = cv2.getTickFrequency() / float(self.get_tick_now() - self.last_tick)
        self.l_fps_history[self.fpsPCounter.next() - 1] = self.fpsP
        self.reset()
        return self.fpsP

    @property
    def avg_fps(self):
        return sum(self.l_fps_history) / float(self.fpsPLen)
