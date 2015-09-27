import os
from os.path import *
from datetime.datetime import *
import logcolors

# create a log colors object
logc = logcolors.LogColors()


def archive(filename):
    pre_move = abs(filename)
    post_move = abs(
        join(filename, str(now().strftime("%Y-%m-%d_%H:%M:%S_")) + basename(filename)))
    try:
        os.rename(pre_move, post_move)
    except:
        print logc.WARN + "[WARN]" + logc.ENDC, "unable to archive", filename
