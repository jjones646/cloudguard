#!/usr/bin/env python
'''
    Help function for getting & setting camera properties.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2


def init_props(cap, config):
    '''
    set and store the real camera properties
    '''
    try:
        if hasattr(cv2, "CAP_PROP_FPS"):
            cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
            config.camera.fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            print("OpenCV not compiled with camera framerate property!")
    except:
        print("Unable to set framerate to {:.1f}!".format(config.camera.fps))
        if hasattr(cv2, "CAP_PROP_FPS"):
            config.camera.fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        print("--  framerate: {}".format(config.camera.fps))

    # set the resolution as specified at the top
    try:
        if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.res[1])
            config.camera.res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        else:
            print("OpenCV not compiled with camera resolution properties!")
    except:
        print("Unable to set resolution to {}x{}!".format(*config.camera.res))
        if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
            config.camera.res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        print("--  resolution: {}x{}".format(*config.camera.res))

    # try to find the fourcc of the attached camera
    try:
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            config.camera.fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        else:
            print("Unable to read camera's codec!")
    except:
        print("Unable to read camera's codec!")
    finally:
        print("--  fourcc: {}".format(config.camera.fourcc))
