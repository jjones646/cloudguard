import os
from os.path import *
from datetime import datetime
from jsonmerge import Merger
import logcolors

# create a log colors object
logc = logcolors.LogColors()

logSchema = {
    "type": "object",
    "properties": {
        "motion": {
            "properties": {
                "motion_count": {"type": "number"},
                "ts": {"type", "date"}
            },
            "mergeStrategy": "append"
        }
    }
}

# archive a file by prefixing a timestamp to the given filename


def archive(filename):
    pre_move = abspath(filename)
    if isfile(pre_move):
        post_move = abspath(join(dirname(pre_move), str(
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")) + basename(filename)))
        try:
            os.rename(pre_move, post_move)
        except:
            print logc.WARN + "[WARN]" + logc.ENDC, "unable to move", pre_move, "to", post_move
    else:
        print logc.WARN + "[WARN]" + logc.ENDC, pre_move, "is not a valid file that can be archived"


def write_log(filename, log_entry):
    filename = abspath(filename)
    if isfile(abspath(filename)):
        needArchive = False
        with open(filename) as f:
            try:
                log_data = json.load(f)
            except ValueError:
                # if file is empty, initialize the
                # json structure & move the current file
                # just to be safe
                log_data = {"motion": []}
                print logc.WARN + "[WARN]" + logc.ENDC, "moving", "could not detect valid json schema in", filename, ", archiving file"
                needArchive = True

        if needArchive:
            archive(filename)

        # append the new timestamp to the current logs
        log_data["motion"].append(log_entry)

        # rewrite the file
        with open(filename, "w") as f:
            json.dump(log_data, f)
    else:
        print logc.WARN + "[WARN]" + logc.ENDC, filename, "is not a valid log file"


# Merge 2 json log files together
# def merge_logs(file1, file2):
