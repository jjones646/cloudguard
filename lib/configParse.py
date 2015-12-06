#!/usr/bin/env python
'''
    Defines classes for parsing the json config file into an object.
'''

# Python 2/3 compatibility
from __future__ import print_function


class ConstConfig(object):

    def __init__(self):
        self.overlay_delim = str(" | ")


class WindowConfig(object):

    def __init__(self, **data):
        self.name = str(data["name"])
        self.enabled = bool(data["enabled"])
        self.overlay_enabled = bool(data["overlay_enabled"])
        self.font_size_stats = int(data["font_size_stats"])
        self.font_size_timestamp = int(data["font_size_timestamp"])
        self.border_x = int(data["border_x"])
        self.border_y = int(data["border_y"])
        self.spacing_y = int(data["spacing_y"])


class CameraConfig(object):

    def __init__(self, **data):
        self.res = tuple(data["resolution"])
        self.fps = float(data["fps"])
        self.rot = int(data["rotation"])
        self.fourcc = None


class ComputingConfig(object):

    def __init__(self, **data):
        self.width = int(data["width"])
        self.bg_sub_thresh = float(data["bg_sub_thresh"])
        self.bg_sub_hist = int(data["bg_sub_hist"])
        self.threading_en = bool(data["threading_enabled"])
        self.face_detection_en = bool(data["face_detection_enabled"])
        self.body_detection_en = bool(data["body_detection_enabled"])
        self.last_motion_timeout = int(data["last_motion_timeout"])


class StorageConfig(object):

    def __init__(self, **data):
        self.save_cloud_crops = bool(data["save_cloud_crops"])
        self.save_local_crops = bool(data["save_local_crops"])
        self.save_local_vids = bool(data["save_local_vids"])
        self.encoding_quality = int(data["encoding_quality"])
        self.min_upload_delay = float(data["min_upload_delay"])


class ConfigObj(object):

    def __init__(self, **config):
        self.window = decode_window_config(config["window"])
        self.camera = decode_camera_config(config["camera"])
        self.computing = decode_computing_config(config["computing"])
        self.storage = decode_storage_config(config["storage"])
        self.const = ConstConfig()

    def show(self):
        print("========================================")
        print("  Config Values:")
        for attr, val in self.__dict__.iteritems():
            for cname, cval in val.__dict__.iteritems():
                ss = attr + "." + cname
                ss = ss.ljust(35)
                print("\t{}=>\t{}".format(ss, cval))
        print("========================================")


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
