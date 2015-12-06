#!/usr/bin/env python
'''
    Help function for getting & setting camera properties.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import imutils


def init_props(cap, config):
    '''
    set and store the real camera properties
    '''

    prop = None
    if imutils.is_cv3():
        if hasattr(cv2, "CAP_PROP_FPS"):
            prop = cv2.CAP_PROP_FPS
    elif imutils.is_cv2():
        if hasattr(cv2.cv, "CV_CAP_PROP_FPS"):
            prop = cv2.cv.CV_CAP_PROP_FPS

    if prop is None:
        print("OpenCV not compiled with camera framerate property!")
    else:
        try:
            cap.set(prop, config.camera.fps)
            config.camera.fps = cap.get(prop)
        except:
            print(
                "Unable to set framerate to {:.1f}!".format(config.camera.fps))
            config.camera.fps = cap.get(prop)
        finally:
            print("--  framerate: {}".format(config.camera.fps))

    # set the resolution as specified at the top
    prop = [None, None]
    if imutils.is_cv3():
        if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
            prop = [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
    elif imutils.is_cv2():
        if hasattr(cv2.cv, "CV_CAP_PROP_FRAME_WIDTH"):
            prop = [cv2.cv.CV_CAP_PROP_FRAME_WIDTH,
                    cv2.cv.CV_CAP_PROP_FRAME_HEIGHT]

    if any(p is None for p in prop):
        print("OpenCV not compiled with camera resolution properties!")
    else:
        try:
            for i in range(1, 2):
                cap.set(prop[i], config.camera.res[i])
                config.camera.res[i] = int(cap.get(prop[i]))
        except:
            print(
                "Unable to set resolution to {}x{}!".format(*config.camera.res))
            config.camera.res = [int(cap.get(p)) for p in prop]
        finally:
            print("--  resolution: {}x{}".format(*config.camera.res))

    # try to find the fourcc of the attached camera
    prop = None
    if imutils.is_cv3():
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            prop = cv2.CAP_PROP_FOURCC
    elif imutils.is_cv2():
        if hasattr(cv2.cv, "CV_CAP_PROP_FOURCC"):
            prop = cv2.cv.CV_CAP_PROP_FOURCC

    if prop is None:
        print("OpenCV not compiled with fourcc property!")
    else:
        try:
            config.camera.fourcc = cap.get(prop)
        except:
            print("Unable to read camera's codec!")
        finally:
            print("--  fourcc: {}".format(config.camera.fourcc))
