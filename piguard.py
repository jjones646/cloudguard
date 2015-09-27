# import the necessary packages
import os
import sys
import time
from os.path import *
from datetime import datetime, timedelta
from math import *
import numpy as np
import json
import argparse
import warnings
import imutils
import cv2
import threading
import logcolors
from logging import *
from imgsearch.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera

# set the name for where the liveview file is saved
liveview_dir = abspath(join(os.getcwd(), 'liveview'))

# make the liveview directory if it doesn't exist
if not exists(liveview_dir):
    os.makedirs(liveview_dir)

liveview_filename = join(liveview_dir, 'liveview.jpg')
liveview_motion_filename = join(liveview_dir, 'liveview_motion.jpg')
liveview_log = join(liveview_dir, 'liveview_log.json')

# create a colors object, enabled by default
logc = logcolors.LogColors()
# logc.disable()

# remove the previous liveview file it one is there
try:
    os.remove(liveview_filename)
    os.remove(liveview_motion_filename)
except OSError:
    pass

# check for existence of the log file
if isfile(liveview_log):
    # archive any current log files by renaming them with a timestamp
    print logc.INFO + "[INFO]" + logc.ENDC, "Archiving", basename(liveview_log)
    archive(liveview_log)

# create this session's logfile
open(liveview_log, "a+").close()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox client
warnings.filterwarnings("ignore")

# load the configuration file, fail if it can't be found
try:
    conf = json.load(open(args["conf"]))
except:
    print logc.FAIL + "[FATAL]" + logc.ENDC, "%s not found...exiting" % args["conf"]
    sys.exit()


# setup the dropbox api if enabled
client = None
if conf["use_dropbox"]:
    from dropbox.client import DropboxOAuth2FlowNoRedirect
    from dropbox.client import DropboxClient

    # connect to dropbox and start the session authorization process
    flow = DropboxOAuth2FlowNoRedirect(
        conf["dropbox_key"], conf["dropbox_secret"])
    print logc.INFO + "[INFO]" + logc.ENDC, "Authorize this application: {}".format(flow.start())
    authCode = raw_input("Enter auth code here: ").strip()

    # finish the authorization and grab the Dropbox client
    (accessToken, userID) = flow.finish(authCode)
    client = DropboxClient(accessToken)
    print logc.OK + "[SUCCESS]" + logc.ENDC, "Dropbox account linked"

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]

rawCapture = PiRGBArray(camera, size=camera.resolution)
rawCapture.truncate(0)  # clear out the buffer before its used

# show OpenCV version information
print logc.INFO + "[INFO]" + logc.ENDC, "OpenCV version:\t%s" % cv2.__version__
# show what resolution we're using
print logc.INFO + "[INFO]" + logc.ENDC, "Camera res.:\t%dx%d" % tuple(conf["resolution"])

# blink the camera's LED to show we're starting up
try:
    # quickly strobe the LED
    for i in range(10):
        camera.led = True
        time.sleep(0.05)
        camera.led = False
        time.sleep(0.05)
    ledState = False
except:
    # LED access requires root privileges, so tell how LED access can be
    # enabled if we can't
    print logc.WARN + "[WARN]" + logc.ENDC, "Insufficient privileges for camera LED control, use sudo for access"
    ledState = True

# allow the camera to warmup
if conf["camera_warmup_time"]:
    # if this is in the config file, use that value
    time.sleep(conf["camera_warmup_time"])
else:
    # default to 3 seconds if not found in config file
    time.sleep(3)

# initialize the average frame
avg = None

# initialize the motion detected counter
motionCounter = 0

# initialize framve timestamp
current_ts = datetime.utcnow()
last_motion_ts = current_ts
last_motion_ts_logged = current_ts
last_upload_ts = current_ts
motion_ts_delta = current_ts

# create a GUI window if enabled
if conf["show_video"]:
    cv2.namedWindow("PiGuard")
    cv2.setWindowProperty(
        "PiGuard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

motionLevel = 0
motionLevel_log = 0
avg_delta_ts = timedelta(0)
# moving average array is the length of our number of triggering frames
# for uploads
moving_average_array = [timedelta()
                        for i in range(int(conf["min_motion_frames"]))]

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # toggle the LED through every frame iteration
    try:
        camera.led = ledState
        ledState = not ledState
    except:
        pass

    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = f.array

    # update the timestamps
    current_ts = datetime.utcnow()
    ts_utc = current_ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    ts_pretty = current_ts.strftime("%A %d %B %Y %I:%M:%S%p")

    # we start out assuming there is no motion in the image
    # the following will only update the detection state if this
    # assumption can be proved wrong
    motion_detected = False
    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print logc.INFO + "[INFO]" + logc.ENDC, "Starting background model"
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(
        frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    _, cnts, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    try:
        if hierarchy.size:
            motionLevel = ceil(2 * sqrt(motionLevel) + float(len(cnts)))
    except AttributeError:
        # decay down to 0
        motionLevel = motionLevel - ((1 / 4) * motionLevel)
        if motionLevel < conf["motion_thresh"]:
            motionLevel = 0.0

    # loop over the contours
    for c in cnts:
        motion_detected = True
        text = "Occupied"
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # write the timestamp & room state over the image
    cv2.putText(
        frame,
        "Room Status: {}".format(text),
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )
    cv2.putText(
        frame,
        ts_pretty,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1
    )

    # check to see if the room is occupied
    if motion_detected:
        last_motion_ts = current_ts
        # check to see if enough time has passed between uploads
        if (current_ts - last_upload_ts).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1

        # see if the number of consecutive frames with motion reaches out
        # threshold
        if motionCounter >= conf["min_motion_frames"]:
            # update the last uploaded timestamp
            last_upload_ts = current_ts

            # reset the motion counter
            motionLevel = motionLevel + motionCounter
            motionCounter = 0

            # see if we should save this locally
            if conf["save_local"]:
                # save image to the liveview frame
                cv2.imwrite(liveview_motion_filename, frame)
                # give some feedback on the console
                print logc.INFO + "[OK]" + logc.OK, "[" + str(ts_utc) + "]", "frame updated"

            # check to see if dropbox sohuld be used
            if conf["use_dropbox"]:
                # write the image to temporary file
                t = TempImage()
                cv2.imwrite(t.path, frame)

                # upload the image to Dropbox and cleanup the tempory image
                print loc.OK + "[UPLOAD]" + loc.ENDC, "{}".format(ts)
                path = "{base_path}/{timestamp}.jpg".format(
                    base_path=conf["dropbox_base_path"], timestamp=ts)
                client.put_file(path, open(t.path, "rb"))
                t.cleanup()

        if conf["log_motion"]:
            # store the timestamp into a json log file
            log_entry = {}
            log_entry["ts"] = str(ts_utc)

            # if (motionLevel_log == 0.0):
            #     log_entry["motion_count"] = 0.0
            #     write_log(liveview_log, log_entry)

            log_entry["motion_count"] = motionLevel_log

            write_log(liveview_log, log_entry)

            motion_ts_delta = current_ts - last_motion_ts_logged
            last_motion_ts_logged = current_ts

            motionLevel_log = motionLevel
            motionLevel_log_last = motionLevel_log

            # delta_ts = current_ts - last_motion_ts
            #delta_ts = current_ts - last_motion_ts_logged
            # print "motion delta:", motion_ts_delta

            # append to front and pop from back
            moving_average_array.append(motion_ts_delta)
            moving_average_array.pop(0)

            # print moving_average_array

            avg_delta_ts = sum(
                moving_average_array, timedelta(0)) / len(moving_average_array)

            print logc.OK + "[OK]" + logc.ENDC, "[" + str(ts_utc) + "]", "moving average:", avg_delta_ts
            # give some feedback on the console
            print logc.INFO + "[OK]" + logc.ENDC, "[" + log_entry["ts"] + "]", "log entry added, motion level:", motionLevel

    # otherwise, the room is not occupied
    else:
        motionCounter = 0
        # setup for next capture
        rawCapture.truncate(0)

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        cv2.imshow("PiGuard", frame)
        key = cv2.waitKey(1) & 0xFF

    if conf["liveview_en"]:
        # save image to the liveview frame
        liveview_tmp = splitext(liveview_filename)[0] + "_tmp.jpg"
        cv2.imwrite(liveview_tmp, frame)

        # switch the name with the last stored frame
        try:
            os.rename(liveview_tmp, liveview_filename)
        except:
            # fallback to removing file first, this is not idea but the best
            # solution at the moment
            os.remove(liveview_filename)
            os.rename(liveview_tmp, liveview_filename)

    # check to see if the running average has fallen to a level indicating
    # that previous movements are no longer in the reference frame
    if (datetime.utcnow() - last_motion_ts_logged) > (10 * avg_delta_ts):
        # write a zero entry motion level to the logs
        if (motionLevel_log != 0.0):
            motionLevel_log = 0.0
            log_entry = {}
            log_entry["motion_count"] = motionLevel_log
            log_entry["ts"] = str(ts_utc)

            write_log(liveview_log, log_entry)
            print logc.FAIL + "[OK]" + logc.ENDC, "[" + str(ts_utc) + "]", "NO MOTION DETECTED"

    # print "delta:", (datetime.utcnow() - last_motion_ts_logged)
    # print "motion delta:", (30 * avg_delta_ts)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
