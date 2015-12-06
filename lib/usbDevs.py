#!/usr/bin/env python
'''
    Defines a basic function that detects the number of connected
    camera/vidio/audio related USB devices that are connected. This
    is simply used for determining any chance of being able to run
    the main program.
'''

# Python 2/3 compatibility
from __future__ import print_function

import os
import usb.core
import usb.util


def num_video_devs():
    '''
    Find out how many cameras are connected.
    '''
    try:
        devs = [usb.core.find(bDeviceClass=0x0e), usb.core.find(
            bDeviceClass=0x10), usb.core.find(bDeviceClass=0xef)]
        devs = [x for x in devs if x is not None]
        dev_s = ""
        if len(devs) == 0:
            print("No USB video devices found!")
            os._exit(130)
        elif len(devs) > 1:
            dev_s = "s"
        print(
            "--  {} audio/video USB device{} detected".format(len(devs), dev_s))
        for d in devs:
            print(
                "--  USB device found at {:04X}:{:04X}".format(d.idVendor, d.idProduct))
    except:
        pass