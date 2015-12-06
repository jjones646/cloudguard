#!/bin/bash

# setup opencv build

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

# go to the opencv source repo
pushd $(readlink -m $SCRIPT_DIR/../external) &> /dev/null
pushd opencv &> /dev/null
OPENCV_CONTRIB_PATH=$(readlink -m ../opencv_contrib/modules)
echo "OpenCV Contrib Path: $OPENCV_CONTRIB_PATH"

if [ -d $OPENCV_CONTRIB_PATH ]; then
    echo "--  OpenCV Contrib modules successfully found"
else
    echo "--  No OpenCV Contrib modules found at $OPENCV_CONTRIB_PATH"
fi

sudo apt-get -y install gstreamer1.0. libgstreamer1.0.

build_base="build"
build_dir=$build_base
counter=0
while [ -d $build_dir ]; do
    let counter+=1
    build_dir=$build_base"$counter"
done
mkdir -p $build_dir && cd $build_dir

cmake \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DWITH_V4L=ON \
-DWITH_LIBV4L=ON \
-DWITH_FFMPEG=OFF \
-DWITH_TBB=ON \
-DWITH_OPENGL=OFF \
-DWITH_CUDA=OFF \
-DWITH_CUFFT=OFF \
-DWITH_CUBLAS=OFF \
-DBUILD_TBB=ON \
-DBUILD_SHARED_LIBS=OFF \
-DENABLE_VFPV3=ON \
-DENABLE_NEON=ON \
-DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_PATH} \
.. > cmake-setup.log

# -DPYTHON_EXECUTABLE=/usr/bin/python2.7 
# -DPYTHON_INCLUDE=/usr/include/python2.7 \
# -DPYTHON_PACKAGES_PATH=/usr/local/lib/python2.7/site-packages \
# -DPYTHON_NUMPY_INCLUDE_DIRS=/usr/local/lib/python2.7/dist-packages/numpy/core/include \

sudo make -j$(nproc)
sudo make install
sudo ldconfig

dirs -c &> /dev/null
exit 0
