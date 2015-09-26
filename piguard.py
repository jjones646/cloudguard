# import the necessary packages
import os, sys, time, datetime
import json
import argparse
import warnings
import imutils
import cv2
import logcolors
from imgsearch.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera

# set the name for where the liveview file is saved
liveview_filename = os.path.join(os.getcwd(), 'liveview', 'liveview.jpg')
liveview_log = os.path.join(os.getcwd(), 'liveview', 'liveview_log.json')

# create a colors object, enabled by default
logc = logcolors.LogColors()
# logc.disable()

# remove the previous liveview file it one is there
try:
    os.remove(liveview_filename)
except OSError:
    pass

# check for existence of the log file
if not os.path.isfile(liveview_log):
    # create file if there isn't a previous one
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
    flow = DropboxOAuth2FlowNoRedirect(conf["dropbox_key"], conf["dropbox_secret"])
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
    # LED access requires root privileges, so tell how LED access can be enabled if we can't
    print logc.WARN + "[WARN]" + logc.ENDC, "Insufficient privileges for camera LED control. use sudo for access"
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
lastUploaded = datetime.datetime.now()
 
# create a GUI window if enabled
if conf["show_video"]:
    cv2.namedWindow("PiGuard")
    cv2.setWindowProperty("PiGuard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
 
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
 
    # update the timestamp
    timestamp = datetime.datetime.now()
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
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
 
    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
    # check to see if the room is occupied
    if text == "Occupied":
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1
 
        # check to see if the number of frames with consistent motion is
        # high enough
        if motionCounter >= conf["min_motion_frames"]:
            # check to see if dropbox sohuld be used
            if conf["use_dropbox"]:
                # write the image to temporary file
                t = TempImage()
                cv2.imwrite(t.path, frame)

                # upload the image to Dropbox and cleanup the tempory image
                print loc.OK + "[UPLOAD]" + loc.ENDC, "{}".format(ts)
                path = "{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
                client.put_file(path, open(t.path, "rb"))
                t.cleanup()

            # update the last uploaded timestamp and reset the motion counter
            lastUploaded = timestamp
            motionCounter = 0

        # see if we should save this locally
        if conf["save_local"]:
            # save image to the liveview frame
            cv2.imwrite(liveview_filename, frame)
            # give some feedback on the console
            print logc.OK + "[SAVE]" + logc.ENDC, "{}".format(ts), "local save"
            # store the timestamp into a json log file
            log_entry = str(ts)
            with open(liveview_log) as f:
                try:
                    log_data = json.loads(f)
                except ValueError:
                    log_data = []   # if file is empty

            # append the new timestamp to the current logs
            log_data["logs"].append({"ts":log_entry})

            # rewrite the file
            with open(liveview_log, "w") as f:
                json.dump(log_data, f)
 
    # otherwise, the room is not occupied
    else:
        motionCounter = 0
        # setup for next capture
        rawCapture.truncate(0)

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        cv2.imshow("PiGuard", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # # if the `q` key is pressed, break from the lop
        # if key == ord("q"):
        # 	break
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 