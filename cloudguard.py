import os, sys, time
import imutils
import cv2
import usb.core, usb.util
import logging
import multiprocessing as mp
import numpy as np
import boto3
import json
from datetime import datetime, timedelta
from os.path import *
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from collections import deque
# local imports
sDir = abspath(dirname(realpath(__file__)))
sys.path.insert(0, join(sDir, "lib"))
from common import clock, draw_str, StatValue, getsize
from persondetect import detectPerson
from facedetect import detectFace, drawFrame
from config_parse import decodeConfig

# set directory for the path to this script
configFn = join(join(sDir, "config.json"))
configFn2 = join(join(sDir, "config2.json"))
vsDir = join(sDir, "vid-streams")
testbench_fn = join(sDir, "testbench_footage_003.mp4")

if not isfile(configFn):
    print("No config file found!")
    sys.exit(1)

# read config file
# try:
with open(configFn, "r") as f:
    config = json.load(f, object_hook=decodeConfig)
# except:
#     print("Error parsing json config file!")
#     sys.exit(2)


# function for resaving the config file
def saveConfig():
    if config is not None:
        try:
            with open(configFn2, 'w') as f:
                json.dump(config, f)
        except:
            print("Error saving config values!")

saveServer = True
saveCrops = saveServer

# params for the text overlaid on the output feed
xBorder = 20
yBorder = 20
ySpacing = 10

fontParams = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=1)

# cap = cv2.VideoCapture(testbench_fn)

# Last Motion Timestamp
LMT = datetime.utcnow()

# Motion Frame Active - starts as false
MFA = False

# timeout for the LMT (seconds)
lmtTo = timedelta(seconds=5)

# find out how many cameras are connected
devs = [usb.core.find(bDeviceClass=0x0e), usb.core.find(bDeviceClass=0x10), usb.core.find(bDeviceClass=0xef)]
devs = [x for x in devs if x is not None]
devsP = ""
if len(devs) == 0:
    print("No USB video devices found!")
    sys.exit(3)
elif len(devs) > 1:
    devsP = "s"
print("--  {} audio/video USB device{} detected".format(len(devs), devsP))
for d in devs:
    print("--  USB device found at {:04X}:{:04X}".format(d.idVendor, d.idProduct))

# this selects the first camera found camera
# cap = cv2.VideoCapture(-1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to connect with camera!")
    sys.exit(4)

# set the framerate as specified at the top
try:
    if cv2.CAP_PROP_FPS is not None:
        cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
    else:
        print("OpenCV not compiled with camera framerate property!")
except:
    print("Unable to set framerate to {:.1f}!".format(config.camera.fps))

config.camera.fps = cap.get(cv2.CAP_PROP_FPS)
print("--  framerate: {}".format(config.camera.fps))

# set the resolution as specified at the top
try:
    if cv2.CAP_PROP_FRAME_WIDTH is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.res[1])
    else:
        print("OpenCV not compiled with camera resolution properties!")
except:
    print("Unable to set resolution to {0}x{1}!".format(*config.camera.res))

config.camera.res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("--  resolution: {0}x{1}".format(*config.camera.res))

contourThresh = 2100 * (float(config.computing.width) / config.camera.res[0])

# display the window if it's enabled in the config
if config.window.en:
    cv2.namedWindow(config.window.name)

    def updateSubHist(x):
        config.computing.bgHist = x

    def updatebgSubThresh(x):
        config.computing.bgThresh = x

    def updateProcessingWidth(x):
        config.computing.width = x

    # create the window
    cv2.namedWindow(config.window.name)

    # trackbars for the window gui
    cv2.createTrackbar('Motion Hist.', config.window.name, 0, 800, updateSubHist)
    cv2.createTrackbar('Motion Thresh.', config.window.name, 0, 40, updatebgSubThresh)
    cv2.createTrackbar('Processing Width', config.window.name, 200, config.camera.res[1], updateProcessingWidth)
    # set initial positions
    cv2.setTrackbarPos('Motion Hist.', config.window.name, config.computing.bgHist)
    cv2.setTrackbarPos('Motion Thresh.', config.window.name, config.computing.bgThresh)
    cv2.setTrackbarPos('Processing Width', config.window.name, config.computing.width)

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(config.computing.bgHist, config.computing.bgThresh)

threadingEn = False
threadN = mp.cpu_count()

pool = ThreadPool(processes=threadN)
pending = deque(maxlen=threadN)

latency = StatValue()
frame_interval = StatValue()
last_frame_time = clock()


def grabFnDate(utc=False):
    if utc is True:
        ts = datetime.utcnow()
    else:
        ts = datetime.now()
    return str(ts).replace(":", "_").replace(" ", "-").replace(".", "_")


def getMotions(f, fMask, thickness=1, color=(170, 170, 170)):
    rectsMot = []
    fRects = np.zeros(f.shape, np.uint8)
    # get contours
    _, cnts, hierarchy = cv2.findContours(fMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < contourThresh:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(fRects, [box], 0, color, thickness)
        rectsMot.append(cv2.boundingRect(c))
    return fRects, rectsMot


def processResponse(q):
    s3 = boto3.resource('s3')
    upWait = 1.5
    lastUp = clock()
    biggestCropArea = 0
    while cap.isOpened():
        # receive the data
        data = q.get()
        f = data["f"]
        rectsSal = data["rectsSal"]
        szScaled = data["szScaled"]
        ts = data["ts"]
        sz = getsize(f)
        rr = (float(sz[0]) / szScaled[0], float(sz[1]) / szScaled[1])
        # rescale the rectangular dimensions to our original resolution
        bx = []
        for x, y, w, h in rectsSal:
            tmp = (x * rr[0], y * rr[1], w * rr[0], h * rr[1])
            bx.append(tuple(int(x) for x in tmp))

        if saveCrops and len(bx) > 0:
            rootP = join(join(sDir, "crop-regions"), grabFnDate())
            xx = tuple((min([min(x[0], x[0] + x[2]) for x in bx]), max([max(x[0], x[0] + x[2]) for x in bx])))
            yy = tuple((min([min(x[1], x[1] + x[3]) for x in bx]), max([max(x[1], x[1] + x[3]) for x in bx])))
            if abs(yy[0] - yy[1]) > 0 and abs(xx[0] - xx[1]) > 0:
                fMask = f[min(yy):max(yy), min(xx):max(xx)]
                cropArea = (max(xx) - min(xx)) * (max(yy) - min(yy))
                fn = rootP + "_regions.jpg"
                if (clock() - lastUp) > upWait:
                    biggestCropArea = 0
                # always send the frames that contain detected people/faces
                if (cropArea > biggestCropArea) or data["numBodies"] > 0 or data["numFaces"] > 0:
                    biggestCropArea = cropArea
                    if saveServer:
                        res, img = cv2.imencode(".jpg", fMask, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                        if res:
                            img = img.tostring()
                            print("uploading frame to s3: {}".format(basename(fn)))
                            print("-- time since last upload: {}s".format(clock() - lastUp))
                            s3.Object("cloudguard-in",
                                      basename(fn)).put(Body=img,
                                                        Metadata={"Content-Type": "Image/jpeg",
                                                                  "Number-Detected-Motion": str(data["numMotion"]),
                                                                  "Number-Detected-Bodies": str(data["numBodies"]),
                                                                  "Number-Detected-Faces": str(data["numFaces"]),
                                                                  "Captured-Timestamp": str(ts),
                                                                  "Captured-Timestamp-Timezone": "UTC"})
                            lastUp = clock()


def processMotionFrame(q, f, tick, ts, mfa=False, rotateAng=False, width=False, gBlur=(9, 9)):
    rectsSal = []
    fCopy = f.copy()
    if rotateAng is not False:
        f = imutils.rotate(f, angle=rotateAng)
    if width is not False:
        f = imutils.resize(f, width=width)
    # blur & bg sub
    fgmask = fgbg.apply(cv2.GaussianBlur(f, gBlur, 0))
    # get our frame outlines
    fRects, rectsMot = getMotions(f, fgmask, thickness=1)
    # return immediately if there's no motion
    rectsSal.extend(rectsMot)
    numMotion = len(rectsMot)
    if True:
    # if numMotion > 0 or mfa is True:
        numBodies = 0
        numFaces = 0
        fBw = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if config.computing.bodyDetectionEn:
            fBody, rectsBody = detectPerson(f, color=(255, 0, 0))
            if len(rectsBody) > 0:
                fRects = cv2.add(fRects, fBody)
                numBodies = len(rectsBody)
                rectsSal.extend(rectsBody)

        if config.computing.faceDetectionEn:
            fFace, rectsFace = detectFace(fBw, color=(0, 255, 0))
            if len(rectsFace) > 0:
                fRects = cv2.add(fRects, fFace)
                numFaces = len(rectsFace)
                rectsSal.extend(rectsFace)

        fRects = imutils.resize(fRects, width=fCopy.shape[1])
        q.put({"f": fCopy.copy(), "ts": ts, "rectsSal": rectsSal, "szScaled": getsize(f), "numMotion": numMotion, "numBodies": numBodies, "numFaces": numFaces})
    return fCopy, fRects, rectsSal, tick, ts


if __name__ == '__main__':
    q = Queue()
    p = Process(target=processResponse, args=(q, ))
    p.start()
    streamId = 0
    vWfn = ["vidStream", ".avi"]
    vwParams = dict(filename=join(vsDir, str("null" + vWfn[1])), fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=config.camera.fps, frameSize=config.camera.res)
    vW = None
    while True:
        while len(pending) > 0 and pending[0].ready():
            frame, fRects, rectsSal, tick, ts = pending.popleft().get()
            latency.update(clock() - tick)
            # overlay the rectangles if motion was detected
            if len(rectsSal) > 0:
                LMT = ts
                sz = frame.shape
                roi = frame[0:sz[0], 0:sz[1]]
                frameMask = cv2.cvtColor(fRects, cv2.COLOR_BGR2GRAY)
                _, frameMask = cv2.threshold(frameMask, 10, 255, cv2.THRESH_BINARY)
                frameBg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(frameMask))
                frameMaskFg = cv2.bitwise_and(fRects, fRects, mask=frameMask)
                frame[0:sz[1], 0:sz[1]] = cv2.add(frameBg, frameMaskFg)

            # overlay a timestamp
            ts = ts.strftime("%A %d %B %Y %I:%M:%S%p (UTC)")
            draw_str(frame, (10, frame.shape[0] - 10), "{}".format(ts), fontScale=0.6, color=(120, 120, 255))
            if threadingEn is False:
                threadDis = 1
            else:
                threadDis = threadN
            # overlay some stats
            statStrings = ["threads: {:<d}".format(threadDis), "res: {1:>d}x{0:<d}".format(*fRects.shape), "latency: {:>6.1f}ms".format(latency.value * 1000), "period: {:>6.1f}ms".format(frame_interval.value * 1000), "fps: {:>5.1f}fps".format(1 / frame_interval.value)]

            txtSz = cv2.getTextSize(statStrings[0], **fontParams)
            xOffset = xBorder
            yOffset = txtSz[0][1] + yBorder
            xDelim = " | "
            tStr = ""
            j = 0
            while True:
                if xOffset != xBorder:
                    statStrings[j] = xDelim + statStrings[j]
                txtSz = cv2.getTextSize(statStrings[j], **fontParams)
                txtSz = txtSz[0]
                xOffset += txtSz[0]
                if xOffset > (sz[1] - xBorder):
                    draw_str(frame, (xBorder, yOffset), tStr, **fontParams)
                    yOffset += txtSz[1] + ySpacing
                    statStrings[j] = statStrings[j][len(xDelim):]
                    tStr = ""
                    xOffset = xBorder
                else:
                    tStr += statStrings[j]
                    j += 1
                if j > len(statStrings) - 1:
                    break

            draw_str(frame, (xBorder, yOffset), tStr, **fontParams)
            if vW is not None:
                vW.write(frame)
            # update the window
            if config.window.en:
                cv2.imshow(config.window.name, frame)

        if (threadingEn is True and len(pending) < threadN) or (threadingEn is False and len(pending) == 0):
            grabbed, frame = cap.read()
            if not grabbed:
                break
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            ts = datetime.utcnow()
            task = pool.apply_async(processMotionFrame, args=(q, frame, t, ts, MFA, config.camera.rot, config.computing.width))
            pending.append(task)

        if config.window.en:
            # refresh the background subtraction parameters
            bgSh = cv2.getTrackbarPos('Motion Hist.', config.window.name)
            bgSt = cv2.getTrackbarPos('Motion Thresh.', config.window.name)
            fgbg.setHistory(bgSh)
            fgbg.setVarThreshold(bgSt)

        # motion is underway
        safeStream = False
        if safeStream:
            if (datetime.utcnow() - LMT) < lmtTo:
                if MFA is False:
                    MFA = True
                    streamId += 1
                    vwParams["filename"] = join(vsDir, str(vWfn[0] + "_{:04d}_".format(streamId) + grabFnDate() + vWfn[1]))
                    vW = cv2.VideoWriter(**vwParams)
            # no activity
            else:
                if MFA is True:
                    MFA = False
                    vW = None

        ch = cv2.waitKey(1)
        # space
        if (ch & 0xff) == ord(' '):
            threadingEn = not threadingEn
        # left arrow key
        if ch == 65361:
            config.camera.rot -= 90
            config.camera.rot %= 360
            saveConfig()
            # right arrow key
        if ch == 65363:
            config.camera.rot += 90
            config.camera.rot %= 360
            saveConfig()
            # escape
        if (ch & 0xff) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

p.terminate()
sys.exit()
