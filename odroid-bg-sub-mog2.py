import numpy as np
import imutils
import time
import os
from os.path import *
import cv2

dimm = (640,480)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimm[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimm[1])

vidStreamName = abspath(join(os.getcwd(), '/home/odroid/Documents/piguard/liveview/vidStream.mp4'))
fourcc = cv2.VideoWriter_fourcc(*'H264')
vidStream = cv2.VideoWriter(vidStreamName, cv2.VideoWriter_fourcc(*'XVID'), 10, dimm)

fgbg = cv2.createBackgroundSubtractorMOG2(200, 14)

startTime = time.time()
frameCount = 0
fps = 0

while(1):
    ret, frame = cap.read()

    if ret == True:
        # rotate
        M = cv2.getRotationMatrix2D((dimm[0]/2, dimm[1]/2), 270, 1.0)
        frame = cv2.warpAffine(frame, M, dimm)
	# resize
	# frame = imutils.resize(frame, width=500)
	# b/w
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_blur = cv2.GaussianBlur(frame, (dimm[0]/30, dimm[0]/30), 0)
        # bg sub
        fgmask = fgbg.apply(frame_blur)

	# get contours
	_, cnts, hierarchy = cv2.findContours(
        fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	frame_show = frame

	# loop over the contours
	for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 3500:
                continue

            # compute the bounding box for the contour, draw it on the frame
            #(x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	    # min area rectangle (rotated)
	    rect = cv2.minAreaRect(c)
	    box = cv2.boxPoints(rect)
	    box = np.int0(box)
	    cv2.drawContours(frame_show, [box], 0, (0,0,255),2)

	# increment the frame counter
	frameCount += 1

	# and print the framerate on our frame
	currtime = time.time()
	runtime = currtime - startTime
	fpsLast = fps
	fps = frameCount / runtime
	if fps > fpsLast:
	    fpsColor = (0,255,0)
	else:
	    fpsColor = (0,0,255)

	cv2.putText(
            frame_show,
            "{:.3f} fps".format(fps),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            fpsColor,
            1
	)

	vidStream.write(frame_show)

	# show the frame
#        cv2.imshow('PiGuard', frame_show)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
