import sys
import imutils
import cv2
import numpy as np
from os.path import *
from common import clock, draw_str

cascade_fn = [abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_upperbody.xml")), abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_frontalface_alt.xml"))]
cascade = [cv2.CascadeClassifier(fn) for fn in cascade_fn]


def detect(frame, cas, sf=1.3, mn=3, ms=(28, 35), mxs=(0, 0)):
    rects = cas.detectMultiScale(frame, scaleFactor=sf, minNeighbors=mn, minSize=ms, maxSize=mxs, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def drawFrame(frame, rects, thickness=1, color=(255, 200, 200)):
    rectsTup = []
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        rectsTup.append((x1, x2, y1, y2))
    return rectsTup


def normalizeFrame(frame):
    frameF = np.zeros(frame.shape, np.uint8)
    if len(frame.shape) > 2:
        frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
        frameF = cv2.cvtColor(frameF, cv2.COLOR_GRAY2BGR)
    return frame, frameF


def detectUppderBody(frame):
    frame, frameF = normalizeFrame(frame)
    rects = detect(frame, cascade[0], sf=1.15, mn=2)
    rects = drawFrame(frameF, rects, thickness=3, color=(130, 130, 255))
    return frameF, rects


def detectFace(frame):
    frame, frameF = normalizeFrame(frame)
    rects = detect(frame, cascade[1], mn=4)
    rects = drawFrame(frameF, rects, thickness=2, color=(0, 255, 0))
    return frameF, rects
