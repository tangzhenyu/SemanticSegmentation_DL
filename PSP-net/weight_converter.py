#!/usr/bin/env python

from __future__ import print_function

import sys
from os.path import splitext
import numpy as np

import caffe

# Not needed because Tensorflow and Caffe do convolution the same way
# Needed for conversion to Theano


def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


weights = {}
assert "prototxt" in splitext(sys.argv[1])[1], "First argument must be caffe prototxt %s" % sys.argv[1]
assert "caffemodel" in splitext(sys.argv[2])[1], "Second argument must be caffe weights %s" % sys.argv[2]
net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
for k, v in net.params.items():
    print ("Layer %s, has %d params." % (k, len(v)))
    if len(v) == 1:
        W = v[0].data[...]
        W = np.transpose(W, (2, 3, 1, 0))
        weights[k] = {"weights": W}
    elif len(v) == 2:
        W = v[0].data[...]
        W = np.transpose(W, (2, 3, 1, 0))
        b = v[1].data[...]
        weights[k] = {"weights": W, "biases": b}
    elif len(v) == 4:  # Batchnorm layer
        k = k.replace('/', '_')
        mean = v[0].data[...]
        variance = v[1].data[...]
        scale = v[2].data[...]
        offset = v[3].data[...]
        weights[k] = {"mean": mean, "variance": variance, "scale": scale, "offset": offset}
    else:
        print("Undefined layer")
        exit()

arr = np.asarray(weights)
weights_name = splitext(sys.argv[2])[0]+".npy"
np.save(weights_name.lower(), arr)
