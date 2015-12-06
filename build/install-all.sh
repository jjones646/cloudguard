#!/bin/bash

# build and install ffmpeg, opencv, and opencv_contrib from source

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
pushd $SCRIPT_DIR &> /dev/null

# update the repo db
sudo apt-get update
sudo apt-get install build-essential linux-headers-$(uname -r)

# update the git submodules
pushd .. &> /dev/null
git submodule update
popd &> /dev/null

# build & install both ffmpeg and opencv with all contrib libraries
./ffmpeg-autobuild.sh
./opencv-autobuild.sh

exit 0
