import sys
import imutils
import cv2
import numpy as np
from os.path import *
from common import clock, draw_str

cascade_fn = abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_frontalface_alt.xml"))
nested_fn  = abspath(join(dirname(realpath(__file__)), "haarcascades/haarcascade_eye.xml"))
cascade = cv2.CascadeClassifier(cascade_fn)
nested = cv2.CascadeClassifier(nested_fn)

def detect(frame, cas):
    rects = cas.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def drawFrame(frame, rects, thickness=1, color=(255,255,0)):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def detectFace(frame):
    frameFrames = np.zeros(frame.shape, np.uint8)
    frameBw= cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    rects = detect(frameBw, cascade)
    drawFrame(frameFrames, rects, thickness=2, color=(0,255,0))

    if not nested.empty():
        for x1, y1, x2, y2 in rects:
            roi = frameBw[y1:y2, x1:x2]
            vis_roi = frameFrames[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            drawFrame(vis_roi, subrects)
    return frameFrames
