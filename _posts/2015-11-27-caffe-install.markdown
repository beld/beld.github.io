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


clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [.build_release/lib/libcaffe.so] Error 1
make: *** Waiting for unfinished jobs....
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_conv_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_lcn_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_lrn_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_pooling_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_relu_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_sigmoid_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_softmax_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn_tanh_layer.o) has no symbols
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib: file: .build_release/lib/libcaffe.a(cudnn.o) has no symbols


ook-Pro:caffe ding$ make runtest
.build_release/tools/caffe
dyld: Library not loaded: @rpath/libcudart.7.5.dylib
  Referenced from: /Users/ding/GitHub/caffe/.build_release/tools/caffe
  Reason: image not found
make: *** [runtest] Trace/BPT trap: 5

Follow these steps to disable SIP:
Restart your Mac.
Before OS X starts up, hold down Command-R and keep it held down until you see an Apple icon and a progress bar. Release. This boots you into Recovery.
From the Utilities menu, select Terminal.
At the prompt type exactly the following and then press Return: csrutil disable
Terminal should display a message that SIP was disabled.
From the  menu, select Restart.

export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib:/Users/ding/anaconda/lib:/usr/local/lib:/usr/lib:/opt/intel/lib:/opt/intel/compilers_and_libraries_2016.1.111/mac/compiler/lib:/opt/intel/compilers_and_libraries_2016.1.111/mac/mkl/lib:/Developer/NVIDIA/CUDA-7.5/lib:$DYLD_FALLBACK_LIBRARY_PATH


SolverTest/0.TestInitTrainTestNets
F1204 02:46:04.125102 1933942784 cudnn_softmax_layer.cpp:15] Check failed: status == CUDNN_STATUS_SUCCESS (1 vs. 0)  CUDNN_STATUS_NOT_INITIALIZED


fatal error: 'Python.h' file not found
