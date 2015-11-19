import os
import sys
import time
import imutils
import cv2
import imutils
import datetime
import multiprocessing as mp
import numpy as np
from os.path import *
from peopledetect import detectPerson
from facedetect import detectFace
from multiprocessing.pool import ThreadPool
from collections import deque
from common import clock, draw_str, StatValue

rez = (640,480)
fps = 15
contourThresh = 2750
bgSubHist = 200
bgSubThresh = 16

# this selects the first camera found camera
cap = cv2.VideoCapture(-1)

# set the framerate as specified at the top
try:
    cap.set(cv2.CAP_PROP_FPS, fps)
except:
    print "Unable to set framerate to {:.3f}!".format(fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print "--  using framerate: {:.3f}".format(fps)

# set the resolution as specified at the top
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rez[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rez[1])
except:
    print "Unable to set resolution to {0}x{0}!".format(*rez)
    rez = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print "--  using resolution: {0}x{1}".format(*rez)

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(bgSubHist, bgSubThresh)

# encoded file object
vidStream_fn = abspath(join(dirname(realpath(__file__)), "liveview/vidStream.avi"))
vidStream = cv2.VideoWriter(vidStream_fn, cv2.VideoWriter_fourcc(*'XVID'), fps, rez)

# video window
cv2.namedWindow('PiGuard')#, cv2.WINDOW_NORMAL)

threadingEn = True
threadN = mp.cpu_count()

pool = ThreadPool(processes=threadN)
pending = deque()

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()

class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

def drawFrame(frame, rects, thickness=1, color=(127,127,127)):
    detections = 0
    frameFrames = np.zeros(frame.shape, np.uint8)
    # get contours
    _, cnts, hierarchy = cv2.findContours(
        rects.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < contourThresh:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(frameFrames, [box], 0, color, thickness)
        detections += 1
    return frameFrames, detections

def processFrame(frame, t0, ts, rotateAng=False, newWidth=False):
    if rotateAng is not False:
        frame = imutils.rotate(frame, angle=rotateAng)
    if newWidth is not False:
        frame = imutils.resize(frame, width=newWidth)

    frameBak = frame.copy()
    #downsample & blur
    cv2.pyrDown(frame.copy(), frame)
    # bg sub
    fgmask = fgbg.apply(frame)
    # get our frame outlines
    frame, det = drawFrame(frame, fgmask, thickness=2)
    # return immediately if there's no motion
    if det != 0:
        frame = cv2.add(frame, detectPerson(frameBak))
        frame = cv2.add(frame, detectFace(frameBak))
    return frameBak, frame, t0, det, ts


while True:
    while len(pending) > 0 and pending[0].ready():
        frame, frameDraw, tt, detected, ts = pending.popleft().get()
        latency.update(clock() - tt)
        if detected > 0:
            # overlay the drawings
            sz = frame.shape
            roi = frame[0:sz[0], 0:sz[1]]
            frameMask = cv2.cvtColor(frameDraw, cv2.COLOR_BGR2GRAY)
            _, frameMask = cv2.threshold(frameMask, 10, 255, cv2.THRESH_BINARY)
            frameBg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(frameMask))
            frameMaskFg = cv2.bitwise_and(frameDraw, frameDraw, mask=frameMask)
            frame[0:sz[1], 0:sz[1]] = cv2.add(frameBg, frameMaskFg)
        # put the time on our frame
        cv2.putText(
                frame,
                "{}".format(ts),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,0,255),
                1
        )
        if threadingEn is False:
            threadDis = 1
        else:
            threadDis = threadN
        # overlay some stats
        draw_str(frame, (20, 20), "threads:        {}".format(threadDis))
        draw_str(frame, (20, 38), "latency:        {:.1f}ms".format(latency.value*1000))
        draw_str(frame, (20, 56), "frame interval: {:.1f}ms".format(frame_interval.value*1000))
        cv2.imshow('PiGuard', frame)

    if len(pending) < threadN:
        ret, frame = cap.read()
        t = clock()
        frame_interval.update(t - last_frame_time)
        last_frame_time = t
        ts = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        if threadingEn:
            task = pool.apply_async(processFrame, args=(frame.copy(), t, ts))
        else:
            task = DummyTask(processFrame(frame, t, ts))

        pending.append(task)

    ch = 0xFF & cv2.waitKey(1)
    if ch == ord(' '):
        threadingEn = not threadingEn
    if ch == 27:
        break

cv2.destroyAllWindows()
