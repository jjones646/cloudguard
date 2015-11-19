import os
import sys
import time
import imutils
import cv2
import imutils
import datetime
import cvfps
import multiprocessing as mp
import numpy as np
from os.path import *
from multiprocessing.pool import ThreadPool
from collections import deque
from common import clock, draw_str, StatValue

dimm = (640,480)
fps = 15

# first camera, with resolution specified above
cap = cv2.VideoCapture(-1)

fpsPre = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS, fps)
fpsNow = cap.get(cv2.CAP_PROP_FPS)

dimmPre = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimm[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimm[1])
dimmNow = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print "Default dimmensions:\t{}".format(dimmPre)
print "Updated dimmensions:\t{}".format(dimmNow)
print "Default FPS:\t\t{:.2f}".format(fpsPre)
print "Updated FPS:\t\t{:.2f}".format(fpsNow)

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(200, 14)

# encoded file object
vidStreamName = abspath(join(os.getcwd(), "/home/jonathan/Documents/piguard/liveview/vidStream.avi"))
vidStream = cv2.VideoWriter(vidStreamName, cv2.VideoWriter_fourcc(*'XVID'), fps, dimm)

# find the size to use for the gaussian blur
blurSz = dimm[0]/30
if blurSz % 2 == 0:
    blurSz += 1
blurSz = (blurSz, blurSz)


class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data


def writeRects(frame, frameRef):
    # get contours
    _, cnts, hierarchy = cv2.findContours(
        frameRef.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 4000:
            continue

        # min area rectangle (rotated)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0,0,255), 2)
    return frame


def processFrame(frame, t0, rotateAng=False, newWidth=False):
    if rotateAng is not False:
        frame = imutils.rotate(frame, angle=rotateAng)
    if newWidth is not False:
        frame = imutils.resize(frame, width=newWidth)
    
    #downsample & blur
    frame_blur = frame
    cv2.pyrDown(frame, frame_blur)
    # bg sub
    fgmask = fgbg.apply(frame_blur)
    # put the time on our frame
    cv2.putText(
            frame,
            "{}".format(t0),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,0,255),
            1
    )
    frame = writeRects(frame, fgmask)
    return frame, t0


threaded_mode = True

cv2.namedWindow('PiGuard', cv2.WINDOW_NORMAL)

threadN = mp.cpu_count()
pool = ThreadPool(processes=threadN)
pending = deque()

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()

fpsTimer = cvfps.cvTimer(length=50, target_fps=fps)

while True:
    while len(pending) > 0 and pending[0].ready():
        res, t0 = pending.popleft().get()
        latency.update(clock() - t0)
        draw_str(res, (20, 20), "threaded:       {}".format(threaded_mode))
        draw_str(res, (20, 40), "threads:        {}".format(threadN))
        draw_str(res, (20, 60), "latency:        {:.1f}ms".format(latency.value*1000))
        draw_str(res, (20, 80), "frame interval: {:.1f}ms".format(frame_interval.value*1000))
        fpsTimer.fps
        draw_str(res, (20, 100),"fps:            {:.2f}".format(fpsTimer.avg_fps))
        cv2.imshow('PiGuard', res)

    if len(pending) < threadN:
        ret, frame = cap.read()
        t = clock()
        frame_interval.update(t - last_frame_time)
        last_frame_time = t
        if threaded_mode:
            task = pool.apply_async(processFrame, args=(frame.copy(), t, 90))
        else:
            task = DummyTask(processFrame(frame, t, 90))

        pending.append(task)

    ch = 0xFF & cv2.waitKey(1)
    if ch == ord(' '):
        threaded_mode = not threaded_mode
    if ch == 27:
        break

cv2.destroyAllWindows()
