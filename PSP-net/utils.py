from __future__ import print_function
import colorsys
import numpy as np

from keras.models import Model


def add_color(img):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in xrange(1, 151):
        img_color[img == i] = to_color(i)
    return img_color


def to_color(category):
    # Maps each category a good distance away
    # from each other on the HSV color space
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v, 1, 1)


# For printing the activations in each layer
# Useful for debugging
def debug(model, data):
    names = [layer.name for layer in model.layers]
    for name in names[:]:
        print_activation(model, name, data)


def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print(layer_name, array_to_str(io))


def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a),
                                   np.max(a), np.mean(a))
