#!/usr/bin/env python
'''
    The top-level file for the CloudGuard.
'''

# Python 2/3 compatibility
from __future__ import print_function

import os
import sys
import imutils
import cv2
import numpy as np
from datetime import datetime, timedelta
from os.path import *
from multiprocessing import Process, Queue, cpu_count
from multiprocessing.pool import ThreadPool
from collections import deque
# from pprint import pprint
# local imports
sDir = abspath(dirname(realpath(__file__)))
sys.path.insert(0, join(sDir, "lib"))
from commonSub import clock, draw_str, StatValue, getsize, grabFnDate
from personDetect import detectPerson
from faceDetect import detectFace
from sysConfig import SysConfig
import camProps

# set directory for the path to this script
configFn = join(join(sDir, "config.json"))
vsDir = join(sDir, "vid-streams")

config = SysConfig(configFn).configblock
config.show()

fontParams = dict(
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=config.window.font_size_stats, thickness=1)

# 'Last Motion Timestamp'
LMT = datetime.utcnow()

# 'Motion Frame Active' - starts as false
MFA = False

# timeout for the LMT (seconds)
lmtTo = timedelta(seconds=config.computing.last_motion_timeout)

# this selects the first camera found camera
cap = cv2.VideoCapture(-1)

if not cap.isOpened():
    print("Unable to connect with camera!")
    os._exit(140)
else:
    camProps.init_props(cap, config)

# compute a contour threshold for indicating whether or not an actual
# object was detected - computed from the frame width for simplicity
contourThresh = int(
    2100 * (float(config.computing.width) / config.camera.res[0]))

# display the window if it's enabled in the config
if config.window.enabled:

        # create the window
    cv2.namedWindow(config.window.name, cv2.WINDOW_NORMAL)

    def updateProcessingWidth(x):
        config.computing.width = x

    # porting all this into opencv2.4 is a real pain, so just cv3
    if imutils.is_cv3():
        def updateSubHist(x):
            config.computing.bg_sub_hist = x

        def updatebgSubThresh(x):
            config.computing.bg_sub_thresh = x

        # trackbars for the window gui
        cv2.createTrackbar(
            'Motion Hist.', config.window.name, 0, 800, updateSubHist)
        cv2.createTrackbar(
            'Motion Thresh.', config.window.name, 0, 40, updatebgSubThresh)
        # set initial positions
        cv2.setTrackbarPos(
            'Motion Hist.', config.window.name, config.computing.bg_sub_hist)
        cv2.setTrackbarPos(
            'Motion Thresh.', config.window.name, int(config.computing.bg_sub_thresh))

    cv2.createTrackbar('Processing Width', config.window.name,
                       200, config.camera.res[1], updateProcessingWidth)
    cv2.setTrackbarPos(
        'Processing Width', config.window.name, config.computing.width)

# background subtractor
if imutils.is_cv3():
    fgbg = cv2.createBackgroundSubtractorMOG2(
        config.computing.bg_sub_hist, config.computing.bg_sub_thresh)
elif imutils.is_cv2():
    fgbg = cv2.BackgroundSubtractorMOG2(
        config.computing.bg_sub_hist, config.computing.bg_sub_thresh)


def getMotions(f, fMask, thickness=1, color=(170, 170, 170)):
    rectsMot = []
    fRects = np.zeros(f.shape, np.uint8)
    # get contours
    if imutils.is_cv3():
        _, cnts, hierarchy = cv2.findContours(
            fMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv2():
        cnts, hierarchy = cv2.findContours(
            fMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < contourThresh:
            continue

        if imutils.is_cv3():
            box = cv2.boxPoints(cv2.minAreaRect(c))
        elif imutils.is_cv2():
            box = cv2.cv.BoxPoints(cv2.minAreaRect(c))

        box = np.int0(box)
        cv2.drawContours(fRects, [box], 0, color, thickness)
        rectsMot.append(cv2.boundingRect(c))
    return fRects, rectsMot


def processResponse(q):
    '''
    This function defines the processes of the "Response Thread" that is
    fully independent from the main thread. This one handles extracting 
    regions from the original (higher-res) frame and uploading the cropped
    out portion locally or to the cloud.
    '''
    import boto3
    s3 = boto3.resource("s3")
    lastUp = clock()
    # initialize our largest detected area to 0
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

        if config.storage.save_cloud_crops or config.storage.save_local_crops:
            if len(bx) > 0:
                xx = tuple(
                    (min([min(x[0], x[0] + x[2]) for x in bx]), max([max(x[0], x[0] + x[2]) for x in bx])))
                yy = tuple(
                    (min([min(x[1], x[1] + x[3]) for x in bx]), max([max(x[1], x[1] + x[3]) for x in bx])))
                # only continue if the area of is not zero
                if abs(yy[0] - yy[1]) > 0 and abs(xx[0] - xx[1]) > 0:
                    fMask = f[min(yy):max(yy), min(xx):max(xx)]
                    cropArea = (max(xx) - min(xx)) * (max(yy) - min(yy))
                    if (clock() - lastUp) > config.storage.min_upload_delay:
                        biggestCropArea = 0
                    # Always send the frames that contain detected people/faces.
                    # If detecting people/faces is disabled in the config, it
                    # is not affected.
                    if (cropArea > biggestCropArea) or data["numBodies"] > 0 or data["numFaces"] > 0:
                        biggestCropArea = cropArea
                        rootP = join(
                            join(sDir, "cropped-regions"), grabFnDate())
                        fn = rootP + "_regions.jpg"
                        res, img = cv2.imencode(
                            ".jpg", fMask, [int(cv2.IMWRITE_JPEG_QUALITY), config.storage.quality])
                        if res and config.storage.save_local_crops:
                            print("saving frame locally: {}".format(rootP))
                            # todo: save locally
                        if res and config.storage.save_cloud_crops:
                            lastUp = clock()
                            img = img.tostring()
                            print(
                                "uploading frame to s3: {}".format(basename(fn)))
                            print(
                                "-- time since last upload: {}s".format(clock() - lastUp))
                            s3.Object("cloudguard-in",
                                      basename(fn)).put(Body=img,
                                                        Metadata={"Content-Type": "Image/jpeg",
                                                                  "Number-Detected-Motion": str(data["numMotion"]),
                                                                  "Number-Detected-Bodies": str(data["numBodies"]),
                                                                  "Number-Detected-Faces": str(data["numFaces"]),
                                                                  "Captured-Timestamp": str(ts),
                                                                  "Captured-Timestamp-Timezone": "UTC"})


def processMotionFrame(q, f, tick, ts, bgm, mfa=False, rotateAng=False, width=False, gBlur=(9, 9)):
    '''
    This function defines the image processing techniques that are applied
    to a new thread when a frame is retreived from the camera.
    '''
    rectsSal = []
    fgmask = None
    fCopy = f.copy()
    if rotateAng is not False and rotateAng != 0:
        f = imutils.rotate(f, angle=rotateAng)
    if width is not False:
        f = imutils.resize(f, width=width)
    # blur & bg sub
    try:
        fgmask = bgm.apply(cv2.GaussianBlur(f, gBlur, 0))
    except Exception as e:
        print(e)
    # get our frame outlines
    fRects, rectsMot = getMotions(f, fgmask, thickness=1)
    rectsSal.extend(rectsMot)
    numMotion = len(rectsMot)

    if True:
        # don't do anything else if there's no motion of any kind detected
        # if numMotion > 0 or mfa is True:
        numBodies = 0
        numFaces = 0
        if config.computing.body_detection_en or config.computing.face_detection_en:
            # generate a histogram equalized bw image if we're doing processing
            # that needs it
            fBw = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            if config.computing.body_detection_en:
                fBody, rectsBody = detectPerson(f, color=(255, 0, 0))
                if len(rectsBody) > 0:
                    fRects = cv2.add(fRects, fBody)
                    numBodies = len(rectsBody)
                    rectsSal.extend(rectsBody)

            if config.computing.face_detection_en:
                fFace, rectsFace = detectFace(fBw, color=(0, 255, 0))
                if len(rectsFace) > 0:
                    fRects = cv2.add(fRects, fFace)
                    numFaces = len(rectsFace)
                    rectsSal.extend(rectsFace)

        fRects = imutils.resize(fRects, width=fCopy.shape[1])
        q.put({"f": fCopy, "ts": ts, "rectsSal": rectsSal, "szScaled": getsize(
            f), "numMotion": numMotion, "numBodies": numBodies, "numFaces": numFaces})

    return f, fRects, rectsSal, tick, ts


if __name__ == '__main__':
    '''
    Main entry point.
    '''
    threadN = cpu_count()
    pool = ThreadPool(processes=threadN)
    # pending = deque(maxlen=threadN)
    pending = deque()
    q = Queue()
    p = Process(target=processResponse, args=(q, ))
    # p.start()
    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    streamId = 0
    vWfn = ["vidStream", ".avi"]
    if imutils.is_cv3():
        fcc = cv2.VideoWriter_fourcc(*"XVID")
    elif imutils.is_cv2():
        fcc = cv2.cv.CV_FOURCC(*"XVID")
    vwParams = dict(filename=join(vsDir, str(
        "null" + vWfn[1])), fourcc=fcc, fps=config.camera.fps, frameSize=config.camera.res)
    vW = None
    while True:
        while len(pending) > 0 and pending[0].ready():
            frame, fRects, rectsSal, tick, ts = pending.popleft().get()
            latency.update(clock() - tick)
            print(rectsSal)
            # overlay the rectangles if motion was detected
            # if len(rectsSal) > 0:
            sz = frame.shape
            if False:
                LMT = ts
                roi = frame[0:sz[0], 0:sz[1]]
                frameMask = cv2.cvtColor(fRects, cv2.COLOR_BGR2GRAY)
                _, frameMask = cv2.threshold(
                    frameMask, 10, 255, cv2.THRESH_BINARY)
                frameBg = cv2.bitwise_and(
                    roi, roi, mask=cv2.bitwise_not(frameMask))
                frameMaskFg = cv2.bitwise_and(fRects, fRects, mask=frameMask)
                frame[0:sz[1], 0:sz[1]] = cv2.add(frameBg, frameMaskFg)

            # overlay a timestamp
            ts = ts.strftime("%A %d %B %Y %I:%M:%S%p (UTC)")
            draw_str(frame, (10, frame.shape[
                     0] - 10), "{}".format(ts), fontScale=config.window.font_size_timestamp, color=(120, 120, 255))

            if config.window.overlay_enabled:
                # the number that we should display for how many threads are
                # currently working
                if config.computing.threading_en:
                    threadDis = threadN
                else:
                    threadDis = 1
                statStrings = ["threads: {:<d}".format(threadDis), "res: {1:>d}x{0:<d}".format(*fRects.shape), "latency: {:>6.1f}ms".format(
                    latency.value * 1000), "period: {:>6.1f}ms".format(frame_interval.value * 1000), "fps: {:>5.1f}fps".format(1 / frame_interval.value)]
                txtSz = cv2.getTextSize(statStrings[0], **fontParams)
                xOffset = config.window.border_x
                yOffset = txtSz[0][1] + config.window.border_y
                tStr = ""
                j = 0
                while True:
                    if xOffset != config.window.border_x:
                        statStrings[
                            j] = config.const.overlay_delim + statStrings[j]
                    txtSz = cv2.getTextSize(statStrings[j], **fontParams)
                    txtSz = txtSz[0]
                    xOffset += txtSz[0]
                    if xOffset > (sz[1] - config.window.border_x):
                        draw_str(
                            frame, (config.window.border_x, yOffset), tStr, **fontParams)
                        yOffset += txtSz[1] + config.window.spacing_y
                        statStrings[j] = statStrings[j][
                            len(config.const.overlay_delim):]
                        tStr = ""
                        xOffset = config.window.border_x
                    else:
                        tStr += statStrings[j]
                        j += 1
                    if j > len(statStrings) - 1:
                        break

                draw_str(
                    frame, (config.window.border_x, yOffset), tStr, **fontParams)

            # update the window if it's enabled
            if config.window.enabled:
                cv2.imshow(config.window.name, frame)

            # write out the frame if our streaming destination object exists
            if vW is not None:
                vW.write(frame)

        if (config.computing.threading_en and len(pending) < threadN) or (not config.computing.threading_en and len(pending) == 0):
            # read a new frame
            grabbed, f = cap.read()
            if not grabbed:
                print("Fatal error attempting to read frame from camera!")
                os._exit(2000)
                break
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            ts = datetime.utcnow()
            task = pool.apply_async(processMotionFrame, args=(
                q, f, t, ts, fgbg, MFA, config.camera.rot, config.computing.width))
            pending.append(task)

        # save a local video stream of detection motion if enabled
        if config.storage.save_local_vids:
            # motion has just begun, generate a new filename and write the
            # first frame
            if (datetime.utcnow() - LMT) < lmtTo:
                if MFA is False:
                    MFA = True
                    streamId += 1
                    vwParams["filename"] = join(
                        vsDir, str(vWfn[0] + "_{:04d}_".format(streamId) + grabFnDate() + vWfn[1]))
                    vW = cv2.VideoWriter(**vwParams)
            # no activity
            else:
                if MFA is True:
                    MFA = False
                    vW = None

        if config.window.enabled:
            # update the slider values to the background model and
            # bind tasks to keystrokes if the window is enabled

            # only with opencv3
            if imutils.is_cv3():
                # refresh the background subtraction parameters
                bgSh = cv2.getTrackbarPos('Motion Hist.', config.window.name)
                bgSt = cv2.getTrackbarPos('Motion Thresh.', config.window.name)
                fgbg.setHistory(bgSh)
                fgbg.setVarThreshold(bgSt)

            ch = cv2.waitKey(2)

            # space
            if (ch & 0xff) == ord(' '):
                config.computing.threading_en = not config.computing.threading_en
            # left arrow key
            if ch == 65361:
                config.camera.rot -= 90
                config.camera.rot %= 360
                # saveConfig()
                # right arrow key
            if ch == 65363:
                config.camera.rot += 90
                config.camera.rot %= 360
                # saveConfig()
                # escape
            if (ch & 0xff) == 27:
                break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    p.terminate()
sys.exit(0)
