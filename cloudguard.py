import os, sys, time
import imutils
import cv2
import usb.core, usb.util
import logging
import multiprocessing as mp
import numpy as np
from datetime import *
from os.path import *
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from collections import deque
# local imports
from common import clock, draw_str, StatValue
from persondetect import detectPerson
from facedetect import detectFace, drawFrame
# from pprint import pprint

windowName = "PiGuard"
rez = (640, 480)
fps = 15
rotation = 0

processingWidth = 320
bgSubHist = 350
bgSubThresh = 8

faceDetectEn = False
fullBodyDetectEn = False

# params for the text overlaid on the output feed
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.45
fontThickness = 1
xBorder = 20
yBorder = 20
ySpacing = 10

# Last Motion Timestamp
LMT = datetime.utcnow()

# Motion Frame Active
MFA = False

# find out how many cameras are connected
devs = [usb.core.find(bDeviceClass=0x0e), usb.core.find(bDeviceClass=0x10), usb.core.find(bDeviceClass=0xef)]
devs = [x for x in devs if x is not None]
devsP = ""
if len(devs) == 0:
    print("No USB video devices found!")
elif len(devs) > 1:
    devsP = "s"
print("--  {} audio/video USB device{} detected".format(len(devs), devsP))
for dev in devs:
    print("--  USB device at {:04X}:{:04X}".format(dev.idVendor, dev.idProduct))

testbench_fn = abspath(join(dirname(realpath(__file__)), "testbench_footage_002.mp4"))

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
rez = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("--  resolution: {0}x{1}".format(*rez))

contourThresh = 2100 * (float(processingWidth) / rez[0])

# video window
cv2.namedWindow(windowName)


def updateSubHist(x):
    bgSubHist = x


def updatebgSubThresh(x):
    bgSubThresh = x


def updateProcessingWidth(x):
    processingWidth = x

# trackbars for the window gui
cv2.createTrackbar('Motion Hist.', windowName, 0, 800, updateSubHist)
cv2.createTrackbar('Motion Thresh.', windowName, 0, 40, updatebgSubThresh)
cv2.createTrackbar('Processing Width', windowName, 200, rez[1], updateProcessingWidth)
# set initial positions
cv2.setTrackbarPos('Motion Hist.', windowName, bgSubHist)
cv2.setTrackbarPos('Motion Thresh.', windowName, bgSubThresh)
cv2.setTrackbarPos('Processing Width', windowName, processingWidth)

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(bgSubHist, bgSubThresh, False)

# encoded file object
vidStream_fn = abspath(join(dirname(realpath(__file__)), "liveview/vidStream.avi"))
vidStream = cv2.VideoWriter(vidStream_fn, cv2.VideoWriter_fourcc(*'XVID'), fps, rez)

threadingEn = True
threadN = mp.cpu_count()

pool = ThreadPool(processes=threadN)
pending = deque(maxlen=threadN)

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()


def getMotions(frame, rects, thickness=1, color=(170, 170, 170)):
    pts = []
    frameF = np.zeros(frame.shape, np.uint8)
    # get contours
    _, cnts, hierarchy = cv2.findContours(rects.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < contourThresh:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(frameF, [box], 0, color, thickness)
        pts.append(cv2.boundingRect(c))
    return frameF, pts


def processResponse(q):
    while True:
        print(type(q.get()))


def processMotionFrame(q, frameI, tick, ts, rotateAng=False, newWidth=False, gBlur=(9, 9)):
    salRects = []
    frameRaw = frameI.copy()
    if rotateAng is not False:
        frameI = imutils.rotate(frameI, angle=rotateAng)
    if newWidth is not False:
        frameI = imutils.resize(frameI, width=newWidth)
    # blur
    frameBlur = cv2.GaussianBlur(frameI, gBlur, 0)
    # bg sub
    fgmask = fgbg.apply(frameBlur)
    # get our frame outlines
    frameRects, rectsMot = getMotions(frameI, fgmask, thickness=1)
    # return immediately if there's no motion
    salRects.extend(rectsMot)
    # if True:
    if len(rectsMot) > 0:
        frameBw = cv2.equalizeHist(cv2.cvtColor(frameI, cv2.COLOR_BGR2GRAY))
        if fullBodyDetectEn:
            frameBody, rectsBody = detectPerson(frameI)
            if len(rectsBody) > 0:
                frameRects = cv2.add(frameRects, frameBody)
                salRects.extend(rectsBody)

        if faceDetectEn:
            frameFace, rectsFace = detectFace(frameBw)
            if len(rectsFace) > 0:
                frameRects = cv2.add(frameRects, frameFace)
                salRects.extend(rectsFace)

        frameRects = imutils.resize(frameRects, width=frameRaw.shape[1])
        LMT = ts
        q.put([frameRaw, ts, salRects])
    else:
        frameRects = None
    return frameRaw, frameRects, tick, ts, salRects


if __name__ == '__main__':
    q = Queue()
    p = Process(target=processResponse, args=(q, )).start()

    while True:
        while len(pending) > 0 and pending[0].ready():
            frame, frameRects, tt, ts, salPts = pending.popleft().get()
            latency.update(clock() - tt)
            # overlay the rectangles if motion was detected
            if len(salPts) > 0 and frameRects is not None:
                sz = frame.shape
                roi = frame[0:sz[0], 0:sz[1]]
                frameMask = cv2.cvtColor(frameRects, cv2.COLOR_BGR2GRAY)
                _, frameMask = cv2.threshold(frameMask, 10, 255, cv2.THRESH_BINARY)
                frameBg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(frameMask))
                frameMaskFg = cv2.bitwise_and(frameRects, frameRects, mask=frameMask)
                frame[0:sz[1], 0:sz[1]] = cv2.add(frameBg, frameMaskFg)

            # overlay a timestamp
            draw_str(frame, (10, frame.shape[0] - 10), "{}".format(ts), fontFace=fontFace, scale=0.6, thickness=1, color=(120, 120, 255))
            if threadingEn is False:
                threadDis = 1
            else:
                threadDis = threadN
            # overlay some stats
            statStrings = ["threads: {:<d}".format(threadDis), "resolution: {1:>d}x{0:<d}".format(*frameRects.shape), "latency: {:>6.1f}ms".format(latency.value * 1000), "period: {:>6.1f}ms".format(frame_interval.value * 1000), "fps: {:>5.1f}fps".format(1 / frame_interval.value)]

            txtSz = cv2.getTextSize(statStrings[0], fontFace, fontScale, fontThickness)
            xOffset = xBorder
            yOffset = txtSz[0][1] + yBorder
            xDelim = " | "
            tStr = ""
            j = 0
            while True:
                if xOffset != xBorder:
                    statStrings[j] = xDelim + statStrings[j]
                txtSz = cv2.getTextSize(statStrings[j], fontFace, fontScale, fontThickness)
                txtSz = txtSz[0]
                xOffset += txtSz[0]
                if xOffset > (sz[1] - xBorder):
                    draw_str(frame, (xBorder, yOffset), tStr, fontFace=fontFace, scale=fontScale, thickness=fontThickness)
                    yOffset += txtSz[1] + ySpacing
                    statStrings[j] = statStrings[j][len(xDelim):]
                    tStr = ""
                    xOffset = xBorder
                else:
                    tStr += statStrings[j]
                    j += 1
                if j > len(statStrings) - 1:
                    break

            draw_str(frame, (xBorder, yOffset), tStr, fontFace=fontFace, scale=fontScale, thickness=fontThickness)
            # update the window
            cv2.imshow(windowName, frame)

        if (threadingEn is True and len(pending) < threadN) or (threadingEn is False and len(pending) == 0):
            grabbed, frame = cap.read()
            if not grabbed:
                break
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            ts = datetime.utcnow().strftime("%A %d %B %Y %I:%M:%S%p (UTC)")
            pWid = cv2.getTrackbarPos('Processing Width', windowName)
            task = pool.apply_async(processMotionFrame, args=(q, frame, t, ts, rotation, pWid))
            pending.append(task)

        # refresh the background subtraction parameters
        bgSh = cv2.getTrackbarPos('Motion Hist.', windowName)
        bgSt = cv2.getTrackbarPos('Motion Thresh.', windowName)
        fgbg.setHistory(bgSh)
        fgbg.setVarThreshold(bgSt)

        ch = cv2.waitKey(1)
        # space
        if (ch & 0xff) == ord(' '):
            threadingEn = not threadingEn
        # left arrow key
        if ch == 65361:
            rotation -= 90
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
            rotation += 90
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
    p.terminate()
