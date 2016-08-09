#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=vggface/solver.prototxt \
    --weights=vggface/VGG_FACE.caffemodel -gpu=0
