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
from upperbodydetect import detectUppderBody
from fullbodydetect import detectBody

rez = (640, 480)
fps = 15
contourThresh = 2500
bgSubHist = 300
bgSubThresh = 14
rotation = 0

faceDetectEn = True
uppderBodyDetectEn = True
fullBodyDetectEn = True

# find out how many cameras are connected
devs = [usb.core.find(bDeviceClass=0x0e), usb.core.find(bDeviceClass=0x10),
        usb.core.find(bDeviceClass=0xef)]
devs = [x for x in devs if x is not None]
devsP = ""
if len(devs) == 0:
    print("No USB video devices found!")
elif len(devs) > 1:
    devsP = "s"
print("--  {} audio/video USB device{} detected".format(len(devs), devsP))
for dev in devs:
    print("--  USB device at {:04X}:{:04X}".format(dev.idVendor, dev.idProduct))

testbench_fn = abspath(join(dirname(realpath(__file__)),
                            "testbench_footage_002.mp4"))

# this selects the first camera found camera
cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture(testbench_fn)

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
pending = deque(maxlen=threadN)

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()


def drawFrame(frame, rects, thickness=1, color=(150, 150, 150)):
    dCount = 0
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
        dCount += 1
    return frameFrames, dCount


def processFrame(frameI, t0, ts, rotateAng=False, newWidth=False):
    shapeOrig = frameI.shape
    if rotateAng is not False:
        frameI = imutils.rotate(frameI, angle=rotateAng)
    if newWidth is not False:
        frameI = imutils.resize(frameI, width=newWidth)
    # blur
    frameBlur = cv2.GaussianBlur(frameI, (9, 9), 0)
    # bg sub
    fgmask = fgbg.apply(frameBlur)
    # get our frame outlines
    frame, det = drawFrame(frameI, fgmask, thickness=1)
    det += 1
    # frameBwHistEq = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frameBwHistEq = frame
    # return immediately if there's no motion
    if det > 0:
        if fullBodyDetectEn is True:
            frameBody, bodyCount, bodyRects = detectBody(frameBwHistEq)  #detectPerson(frameI)
            if bodyCount != 0:
                frame = cv2.add(frame, frameBody)
        if uppderBodyDetectEn is True:
            frameUB, uBodyCount, uBodyRects = detectUppderBody(frameBwHistEq)
            if uBodyCount != 0:
                frame = cv2.add(frame, frameUB)
        if faceDetectEn is True:
            faceF, faceCount, faceRects = detectFace(frameBwHistEq)
            if faceCount != 0:
                frame = cv2.add(frame, faceF)

        print(bodyCount, uBodyCount, faceCount)
    return frameI, frame, t0, det, ts, shapeOrig


while True:
    while len(pending) > 0 and pending[0].ready():
        frame, frameDraw, tt, detected, ts, origShape = pending.popleft().get()
        frame = imutils.resize(frame, width=origShape[1])
        frameDraw = imutils.resize(frameDraw, width=origShape[1])
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
            frame,
            "{}".format(ts),
            (11, frame.shape[0] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            lineType=cv2.LINE_AA)
        cv2.putText(frame,
                    "{}".format(ts),
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    lineType=cv2.LINE_AA)

        if threadingEn is False:
            threadDis = 1
        else:
            threadDis = threadN
        # overlay some stats
        draw_str(frame, (20, 20), "threads:{:>13d}".format(threadDis))
        draw_str(frame, (20, 38),
                 "resolution:{1:>12d}x{0:<6d}".format(*frameDraw.shape))
        draw_str(frame, (20, 56),
                 "latency:{:>14.1f}ms".format(latency.value * 1000))
        draw_str(frame, (20, 74),
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
        task = pool.apply_async(
            processFrame,
            args=(frame, t, ts, rotation, int(frame.shape[1] / 2)))
        pending.append(task)

    ch = cv2.waitKey(1)
    # space
    if (ch & 0xff) == ord(' '):
        threadingEn = not threadingEn
    # left arrow key
    if ch == 65361:
        rotation += 90
        rotation %= 360
    if ch == 49:
        fullBodyDetectEn = not fullBodyDetectEn
        print("Full body detection: {}".format(fullBodyDetectEn))
    if ch == 50:
        uppderBodyDetectEn = not uppderBodyDetectEn
        print("Upper body detection: {}".format(uppderBodyDetectEn))
    if ch == 51:
        faceDetectEn = not faceDetectEn
        print("Face detection: {}".format(faceDetectEn))
    # up arrow key
    if ch == 65362:
        contourThresh += 250
        print("New contour threshold: {}".format(contourThresh))
    # right arrow key
    if ch == 65363:
        rotation -= 90
        rotation %= 360
    # down arrow key
    if ch == 65364:
        contourThresh -= 250
        print("New contour threshold: {}".format(contourThresh))
    # escape
    if (ch & 0xff) == 27:
        break

cap.release()
cv2.destroyAllWindows()
