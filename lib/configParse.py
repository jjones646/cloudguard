#!/usr/bin/env python
'''
    Defines classes for parsing the json config file into an object.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import textwrap


class ConstConfig(object):

    def __init__(self):
        self.overlay_delim = str(" | ")
        self.__comment__ = "These are constants set in the code itself."


class WindowConfig(object):

    def __init__(self, **data):
        self.name = str(data["name"]["value"])
        self.enabled = bool(data["enabled"]["value"])
        self.overlay_enabled = bool(data["overlay_enabled"]["value"])
        self.font_size_stats = float(data["font_size_stats"]["value"])
        self.font_size_timestamp = float(data["font_size_timestamp"]["value"])
        self.border_x = int(data["border_x"]["value"])
        self.border_y = int(data["border_y"]["value"])
        self.spacing_y = int(data["spacing_y"]["value"])
        self._font_params = dict(
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_size_stats, thickness=1)
        if "__comment__" in data:
            self.__comment__ = str(data["__comment__"])


class CameraConfig(object):

    def __init__(self, **data):
        self.res = tuple(data["resolution"]["value"])
        self.fps = float(data["fps"]["value"])
        self.rot = int(data["rotation"]["value"])
        self.fourcc = str(data["rotation"]["value"])
        if "__comment__" in data:
            self.__comment__ = str(data["__comment__"])


class ComputingConfig(object):

    def __init__(self, **data):
        self.width = int(data["width"]["value"])
        self.bg_sub_thresh = float(data["bg_sub_thresh"]["value"])
        self.bg_sub_hist = int(data["bg_sub_hist"]["value"])
        self.threading_en = bool(data["threading_enabled"]["value"])
        self.face_detection_en = bool(data["face_detection_enabled"]["value"])
        self.body_detection_en = bool(data["body_detection_enabled"]["value"])
        self.last_motion_timeout = int(data["last_motion_timeout"]["value"])
        self.learning_rate = float(data["learning_rate"]["value"])
        if "__comment__" in data:
            self.__comment__ = str(data["__comment__"])


class StorageConfig(object):

    def __init__(self, **data):
        self.save_cloud_crops = bool(data["save_cloud_crops"]["value"])
        self.save_local_crops = bool(data["save_local_crops"]["value"])
        self.save_local_vids = bool(data["save_local_vids"]["value"])
        self.encoding_quality = int(data["encoding_quality"]["value"])
        self.min_upload_delay = float(data["min_upload_delay"]["value"])
        if "__comment__" in data:
            self.__comment__ = str(data["__comment__"])


class ConfigObj(object):

    def __init__(self, **config):
        self.window = decode_window_config(config["window"])
        self.camera = decode_camera_config(config["camera"])
        self.computing = decode_computing_config(config["computing"])
        self.storage = decode_storage_config(config["storage"])
        self.const = ConstConfig()
        if "__comment__" in config:
            self.__comment__ = str(config["__comment__"])

    def show(self):
        ww, hh = get_terminal_size()
        print("="*ww)
        print("  Config Values:")
        for attr, val in self.__dict__.iteritems():
            cfg_copy = val
            if hasattr(cfg_copy, "__comment__"):
                cfg_copy.__comment__ = textwrap.wrap(
                    cfg_copy.__comment__, width=ww-10)
                for line in cfg_copy.__comment__:
                    print("\t{}".format(line))
            for cname, cval in cfg_copy.__dict__.iteritems():
                # only show configs that don't start with an underscore
                if cname[:1] != "_":
                    ss = attr + "." + cname
                    ss = ss.ljust(35)
                    print("\t   {}=>\t{}".format(ss, cval))
            print("")
        print("="*ww)


def decode_window_config(obj):
    if "__type__" in obj and obj["__type__"] == "window_config":
        return WindowConfig(**obj)
    return obj


def decode_camera_config(obj):
    if "__type__" in obj and obj["__type__"] == "camera_config":
        return CameraConfig(**obj)
    return obj


def decode_computing_config(obj):
    if "__type__" in obj and obj["__type__"] == "computing_config":
        return ComputingConfig(**obj)
    return obj


def decode_storage_config(obj):
    if "__type__" in obj and obj["__type__"] == "storage_config":
        return StorageConfig(**obj)
    return obj


def json2config(obj):
    if "__type__" in obj and obj["__type__"] == "config_block":
        return ConfigObj(**obj["config"])
    return obj


def get_terminal_size():
    '''
    Determines and returns the current terminal's dimensions.
    '''
    import os
    env = os.environ

    def ioctl_gwinsz(fd):
        try:
            import fcntl
            import termios
            import struct
            cr = struct.unpack(
                'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return
        return cr
    cr = ioctl_gwinsz(0) or ioctl_gwinsz(1) or ioctl_gwinsz(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_gwinsz(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
    return int(cr[1]), int(cr[0])
