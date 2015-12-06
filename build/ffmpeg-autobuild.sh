#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

pushd $(readlink -m $SCRIPT_DIR/../external/ffmpeg) &> /dev/null

./configure \
--prefix=/usr/local \
--enable-libxvid \
--enable-libx264 \
--enable-libv4l2 \
--enable-pic \
--enable-shared \
--enable-gpl \
--enable-nonfree \
--disable-htmlpages \
--disable-podpages \
--disable-txtpages

# --enable-libopenh264
# --enable-libopencv

make -j$(nproc)
sudo make install
sudo ldconfig

dirs -c &> /dev/null
exit 0
