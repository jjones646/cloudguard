import cv2
import imutils
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def drawFrame(frame, rects, thickness=1, color=(255,0,0)):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(frame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), color, thickness)

def detectPerson(frame):
    found_filtered = []
    frameFrames = np.zeros(frame.shape, np.uint8)
    found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(35,35), scale=1.05)
    
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r,q):
                break
        else:
            found_filtered.append(r)

    drawFrame(frameFrames, found, color=(255,127,0))
    drawFrame(frameFrames, found_filtered, thickness=3)
    return frameFrames
