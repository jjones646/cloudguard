'''
    Defines classes for the json config file.
'''

import json


class windowConfig(object):
    def __init__(self, **data):
        self.name = str(data["title"])
        self.en = bool(data["enabled"])


class cameraConfig(object):
    def __init__(self, **data):
        self.res = tuple(data["resolution"])
        self.fps = float(data["fps"])
        self.rot = int(data["rotation"])


class computingConfig(object):
    def __init__(self, **data):
        self.width = int(data["width"])
        self.bgThresh = int(data["sub_thresh"])
        self.bgHist = int(data["sub_hist"])
	self.faceDetectionEn = bool(data["face_detection"])
	self.bodyDetectionEn = bool(data["body_detection"])


class cloudConfig(object):
    def __init__(self, **data):
        self.upload = bool(data["upload"])
        self.saveLocal = bool(data["save_local"])
        self.minHoldoff = float(data["min_upload_seconds"])
        self.minFrames = int(data["min_motion_frames"])


class Config(object):
    def __init__(self, **config):
        self.window = decodeWindowConfig(config["window"])
        self.camera = decodeCameraConfig(config["camera"])
        self.computing = decodeComputingConfig(config["computing"])
        self.cloud = decodeCloudConfig(config["cloud"])


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


def decodeCloudConfig(obj):
    if "__type__" in obj and obj["__type__"] == "cloud_config":
        return cloudConfig(**obj)
    return obj


def decodeConfig(obj):
    if "__type__" in obj and obj["__type__"] == "config_block":
        return Config(**obj["config"])
    return obj
