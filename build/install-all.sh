#!/bin/bash

# build and install ffmpeg, opencv, and opencv_contrib from source

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
pushd $SCRIPT_DIR &> /dev/null

sudo apt-get update

./ffmpeg-autobuild.sh
./opencv-autobuild.sh

exit 0
