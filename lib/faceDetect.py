'''
    Basic wrapper functions for using OpenCV's Cascade Classifier
    support for classifying and detecting faces in frames.
'''

# Python 2/3 compatibility
from __future__ import print_function

# import sys
# import imutils
import cv2
import numpy as np
from os.path import join, dirname, realpath, abspath

cascade_fn = [abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_upperbody.xml")),
              abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_frontalface_alt.xml"))]
cascade = [cv2.CascadeClassifier(fn) for fn in cascade_fn]


def detect(f, cas, sf=1.3, mn=3, ms=(28, 35), mxs=(0, 0)):
    rects = cas.detectMultiScale(
        f, scaleFactor=sf, minNeighbors=mn, minSize=ms, maxSize=mxs, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def drawFrame(f, rects, thickness=1, color=(255, 200, 200)):
    rectsTup = []
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(f, (x1, y1), (x2, y2), color, thickness)
        # convert to x, y, w, d
        rectsTup.append((x1, y1, abs(x2 - x1), abs(y2 - y1)))
    return rectsTup


def normalizeFrame(f):
    fRect = np.zeros(f.shape, np.uint8)
    if len(f.shape) > 2:
        f = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    else:
        fRect = cv2.cvtColor(fRect, cv2.COLOR_GRAY2BGR)
    return f, fRect


def detectUppderBody(f):
    f, fRect = normalizeFrame(f)
    rects = detect(f, cascade[0], sf=1.15, mn=2)
    rects = drawFrame(fRect, rects, thickness=3, color=(130, 130, 255))
    return fRect, rects


def detectFace(f, thickness=2, color=(0, 255, 0)):
    f, fRect = normalizeFrame(f)
    rects = detect(f, cascade[1], mn=4)
    rects = drawFrame(fRect, rects, thickness=thickness, color=color)
    return fRect, rects
