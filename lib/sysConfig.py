#!/usr/bin/env python
'''
    Defines classes for parsing the json config file into an object.
'''

# Python 2/3 compatibility
from __future__ import print_function

import os.path
import json
from configParse import json2config


class SysConfig(object):

    def __init__(self, fn):
        self.fn = fn
        self.config = self.load()

    @property
    def configblock(self):
        return self.config

    def save(self):
        if self.config is not None:
            try:
                with open(self.fn, 'w') as f:
                    json.dump(self.fn, f)
            except:
                print("Error saving config values!")
                os._exit(50)

    def load(self):
        if not os.path.isfile(self.fn):
            print("No config file found!")
            os._exit(100)
        try:
            with open(self.fn, "r") as f:
                cfg = json.load(f, object_hook=json2config)
            return cfg
        except IOError as e:
            print(e)
            os._exit(101)
        except Exception as e:
            print("Error parsing json config file!")
            print(e)
            os._exit(102)

    def show(self):
        self.config.show()
