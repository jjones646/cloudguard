import os, sys, time, datetime
import imutils
import cv2
import usb.core, usb.util
import multiprocessing as mp
import numpy as np
from os.path import *
from multiprocessing.pool import ThreadPool
from collections import deque
# local imports
from common import clock, draw_str, StatValue
from peopledetect import detectPerson
from facedetect import detectFace

rez = (640, 480)
fps = 10
contourThresh = 3000
bgSubHist = 250
bgSubThresh = 14
rotation = 0

# find out how many cameras are connected
devs = [usb.core.find(bDeviceClass=0x0e), usb.core.find(bDeviceClass=0x10),
        usb.core.find(bDeviceClass=0xef)]
devs = [x for x in devs if x is not None]
if len(devs) == 0:
    print("No USB video devices found!")
elif len(devs) > 1:
    devsP = "s"
else:
    devsP = ""
print("--  {} audio/video USB device{} detected".format(len(devs), devsP))
for dev in devs:
    print("--  USB device at {:04X}:{:04X}".format(dev.idVendor, dev.idProduct))

# this selects the first camera found camera
# cap = cv2.VideoCapture(-1)
testbench_fn = abspath(join(dirname(realpath(__file__)),
                            "testbench_footage_001.mp4"))
cap = cv2.VideoCapture(testbench_fn)
if not cap.isOpened():
    print("Unable to connect with camera!")

# set the framerate as specified at the top
try:
    cap.set(cv2.CAP_PROP_FPS, fps)
except:
    print("Unable to set framerate to {:.1f}!".format(fps))
    fps = cap.get(cv2.CAP_PROP_FPS)
print("--  framerate: {}".format(fps))

# set the resolution as specified at the top
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rez[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rez[1])
except:
    print("Unable to set resolution to {0}x{1}!".format(*rez))
    rez = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
           cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("--  resolution: {0}x{1}".format(*rez))

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(bgSubHist, bgSubThresh)

# encoded file object
vidStream_fn = abspath(join(dirname(realpath(__file__)), "liveview/vidStream.avi"))
vidStream = cv2.VideoWriter(vidStream_fn, cv2.VideoWriter_fourcc(*'XVID'), fps,
                            rez)

# video window
cv2.namedWindow('PiGuard')

threadingEn = True
threadN = mp.cpu_count()

pool = ThreadPool(processes=threadN)
pending = deque()

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()


def drawFrame(frame, rects, thickness=1, color=(127, 127, 127)):
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
    det += 1
    # return immediately if there's no motion
    if det != 0:
        frame = cv2.add(frame, detectPerson(frameBak))

        faceF, faceCount, faceRects = detectFace(frameBak)
        if faceCount != 0:
            frame = cv2.add(frame, faceF)
            print("{:d} faces detected").format(faceCount)

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
            frameBg = cv2.bitwise_and(roi,
                                      roi,
                                      mask=cv2.bitwise_not(frameMask))
            frameMaskFg = cv2.bitwise_and(frameDraw, frameDraw, mask=frameMask)
            frame[0:sz[1], 0:sz[1]] = cv2.add(frameBg, frameMaskFg)
        # put the time on our frame
        cv2.putText(
            frame, "{}".format(ts), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        if threadingEn is False:
            threadDis = 1
        else:
            threadDis = threadN
        # overlay some stats
        draw_str(frame, (20, 20), "threads:{:>13d}".format(threadDis))
        draw_str(frame, (20, 38),
                 "latency:{:>14.1f}ms".format(latency.value * 1000))
        draw_str(frame, (20, 56),
                 "frame interval:{:>7.1f}ms  ({:<5.1f}fps)".format(
                     frame_interval.value * 1000, 1 / frame_interval.value))
        cv2.imshow('PiGuard', frame)

    if (threadingEn is True and len(pending) < threadN) or (
            threadingEn is False and len(pending) == 0):
        _, frame = cap.read()
        t = clock()
        frame_interval.update(t - last_frame_time)
        last_frame_time = t
        ts = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        task = pool.apply_async(processFrame,
                                args=(frame.copy(), t, ts, rotation))
        pending.append(task)

    ch = cv2.waitKey(1)
    # space
    if (ch & 0xff) == ord(' '):
        threadingEn = not threadingEn
    # left arrow key
    if ch == 65361:
        rotation += 90
        rotation %= 360
    # up arrow key
    if ch == 65362:
        contourThresh += 250
    # right arrow key
    if ch == 65363:
        rotation -= 90
        rotation %= 360
    # down arrow key
    if ch == 65364:
        contourThresh -= 250
    # escape
    if (ch & 0xff) == 27:
        break

cap.release()
cv2.destroyAllWindows()
