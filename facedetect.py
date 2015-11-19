import sys
import imutils
import cv2
import numpy as np
from os.path import *
from common import clock, draw_str

cascade_fn = abspath(join(dirname(realpath(__file__)),
                          "haarcascades/haarcascade_frontalface_alt.xml"))
# nested_fn = abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_eye.xml"))

cascade = cv2.CascadeClassifier(cascade_fn)
# nested = cv2.CascadeClassifier(nested_fn)


def detect(frame, cas, sf=1.3, mn=3, ms=(28, 35), mxs=(0, 0)):
    rects = cas.detectMultiScale(frame,
                                 scaleFactor=sf,
                                 minNeighbors=mn,
                                 minSize=ms,
                                 maxSize=mxs,
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def drawFrame(frame, rects, thickness=1, color=(255, 255, 0)):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def detectFace(frame):
    frameFrames = np.zeros(frame.shape, np.uint8)
    frameBw = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    rects = detect(frame, cascade, mn=4)
    drawFrame(frameFrames, rects, thickness=2, color=(0, 255, 0))
    return frameFrames, len(rects), rects
