import cv2
import imutils
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def drawFrame(f, rects, thickness=1, color=(255, 0, 0)):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        xPad, yPad = int(0.2 * w), int(0.1 * h)
        cv2.rectangle(f, (x + xPad, y + yPad), (x + w - xPad, y + h - yPad), color, thickness)


def detectPerson(f, color=(255, 127, 127), thickness=3):
    foundFiltered = []
    fRects = np.zeros(f.shape, np.uint8)
    found, w = hog.detectMultiScale(f, winStride=(12, 12), padding=(15, 5), scale=1.02)
    for i, r in enumerate(found):
        for j, q in enumerate(found):
            if i != j and inside(r, q):
                break
        else:
            foundFiltered.append(r)
    drawFrame(fRects, found, color=color)
    drawFrame(fRects, foundFiltered, thickness=thickness)
    return fRects, foundFiltered
