---
layout:     post
title:      "Caffe Installation"
subtitle:   " \"caffe is a bitch\""
date:       2015-11-29 23:00:00
author:     "Beld"
header-img: "img/post-bg-2015.jpg"
tags:
    - 生活
    - Life
---

>This is an empty article, waiting to be finished

Error lists:

make all -j8
PROTOC src/caffe/proto/caffe.proto
make: protoc: No such file or directory
make: *** [.build_release/src/caffe/proto/caffe.pb.cc] Error 1
make: *** Waiting for unfinished jobs....

brew link protobuf
Linking /usr/local/Cellar/protobuf/2.6.1...
Error: Could not symlink lib/python2.7/site-packages/homebrew-protobuf.pth
/usr/local/lib/python2.7/site-packages is not writable.

sudo chown -R Ding /usr/local/lib/python2.7/site-packages

'hdf5.h' file not found #include "hdf5.h"
brew install hdf5

make all -j8
src/caffe/data_transformer.cpp:2:10: fatal error: 'opencv2/core/core.hpp' file not found

Package opencv was not found in the pkg-config search path.
Perhaps you should add the directory containing 'opencv.pc'
to the PKG_CONFIG_PATH environment variable
No package 'opencv' found

mdfind opencv.pc
/usr/local/Cellar/opencv/2.4.12/lib/pkgconfig/opencv.pc
/usr/local/Cellar/opencv3/3.0.0/lib/pkgconfig/opencv.pc
export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.12/lib/pkgconfig/opencv.pc


pkg-config --libs --cflags opencv | sed 's/libtbb\.dylib/tbb/' -I/usr/local/Cellar/opencv/2.4.12/include/opencv -I/usr/local/Cellar/opencv/2.4.12/include /usr/local/Cellar/opencv/2.4.12/lib/libopencv_calib3d.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_contrib.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_core.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_features2d.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_flann.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_gpu.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_highgui.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_imgproc.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_legacy.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_ml.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_nonfree.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_objdetect.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_ocl.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_photo.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_stitching.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_superres.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_ts.a /usr/local/Cellar/opencv/2.4.12/lib/libopencv_video.dylib /usr/local/Cellar/opencv/2.4.12/lib/libopencv_videostab.dylib -ltbb



export OpenCV_DIR=/usr/local/Cellar/opencv/2.4.12/
