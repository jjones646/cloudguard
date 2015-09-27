import os
from os.path import *
from datetime import datetime
import logcolors

# create a log colors object
logc = logcolors.LogColors()


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
