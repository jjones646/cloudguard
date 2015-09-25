import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import datetime
import cv2

# get the working directory path for saving files
workPath = os.getcwd()

# initialize camera & reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# warmup the camera
time.sleep(0.1)

# grab the current time
timestamp = datetime.datetime.now()

# now, get the image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

image = cv2.flip(image, 0)

cv2.putText(image, "Yoooo!!", (250, 700), cv2.FONT_HERSHEY_COMPLEX, 4, cv2.FILLED)

ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
cv2.putText(image, ts, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

cv2.imwrite(os.path.join(workPath, 'opencv-test.jpg'), image)
###camera.capture( os.path.abspath(os.path.join(workPath, 'test-image.jpg')) )

