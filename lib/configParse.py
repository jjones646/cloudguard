#!/usr/bin/env python
'''
    Defines classes for parsing the json config file into an object.
'''

import json
from pprint import pprint


class constConfig(object):
    def __init__(self):
        self.overlay_delim = str(" | ")


class windowConfig(object):
    def __init__(self, **data):
        self.name = str(data["name"])
        self.enabled = bool(data["enabled"])
        self.overlay_enabled = bool(data["overlay_enabled"])
        self.border_x = int(data["border_x"])
        self.border_y = int(data["border_y"])
        self.spacing_y = int(data["spacing_y"])


class cameraConfig(object):
    def __init__(self, **data):
        self.res = tuple(data["resolution"])
        self.fps = float(data["fps"])
        self.rot = int(data["rotation"])


class computingConfig(object):
    def __init__(self, **data):
        self.width = int(data["width"])
        self.bg_sub_thresh = float(data["bg_sub_thresh"])
        self.bg_sub_hist = int(data["bg_sub_hist"])
        self.threading_en = bool(data["threading_enabled"])
        self.face_detection_en = bool(data["face_detection_enabled"])
        self.body_detection_en = bool(data["body_detection_enabled"])


class storageConfig(object):
    def __init__(self, **data):
        self.save_cloud_crops = bool(data["save_cloud_crops"])
        self.save_local_crops = bool(data["save_local_crops"])
        self.save_local_vids = bool(data["save_local_vids"])
        self.encoding_quality = int(data["encoding_quality"])
        self.min_upload_delay = float(data["min_upload_delay"])


class Config(object):
    def __init__(self, **config):
        self.window = decodeWindowConfig(config["window"])
        self.camera = decodeCameraConfig(config["camera"])
        self.computing = decodeComputingConfig(config["computing"])
        self.storage = decodeStorageConfig(config["storage"])
        self.const = constConfig()


def decodeWindowConfig(obj):
    if "__type__" in obj and obj["__type__"] == "window_config":
        return windowConfig(**obj)
    return obj


def decodeCameraConfig(obj):
    if "__type__" in obj and obj["__type__"] == "camera_config":
        return cameraConfig(**obj)
    return obj


def decodeComputingConfig(obj):
    if "__type__" in obj and obj["__type__"] == "computing_config":
        return computingConfig(**obj)
    return obj


def decodeStorageConfig(obj):
    if "__type__" in obj and obj["__type__"] == "storage_config":
        return storageConfig(**obj)
    return obj


def decodeConfig(obj):
    if "__type__" in obj and obj["__type__"] == "config_block":
        return Config(**obj["config"])
    return obj


def dump_config(obj):
    print("========================================")
    print("  Config Values:")
    for attr, v in obj.__dict__.iteritems():
        for aa, vv in v.__dict__.iteritems():
            ss = attr + "." + aa
            ss = ss.ljust(35)
            print("\t{}=>\t{}".format(ss, vv))
    print("========================================")
