import numpy as np
import cv2
import sys
import getopt
from os.path import *
# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


cascade_fn = abspath(join(realpath(__file__), "haarcascades/haarcascade_frontalface_alt.xml"))
nested_fn  = abspath(join(realpath(__file__), "haarcascades/haarcascade_eye.xml"))

cap = cv2.VideoCapture(-1)

cascade = cv2.CascadeClassifier(cascade_fn)
nested = cv2.CascadeClassifier(nested_fn)

cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

while True:
    ret, frame = cap.read()
    frame = imutils.rotate(frame, angle=90)

    frameBw= cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    t = clock()
    rects = detect(frameBw, cascade)
    vis = frame.copy()
    draw_rects(vis, rects, (0,255,0))

    if not nested.empty():
        for x1, y1, x2, y2 in rects:
            roi = frameBw[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            draw_rects(vis_roi, subrects, (255, 0, 0))

    dt = clock() - t

    draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
    cv2.imshow('PiGuard - Face Detect', vis)

    if 0xFF & cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()
