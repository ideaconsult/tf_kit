#####
# Variational Auto-Encoder based dimensionality reduction
# File: Some general tools, including data loading
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev

from __future__ import print_function
import numpy as np
import gzip
import os
import re
import sys


try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def nn_parse_architecture(arch_str, func_dict):
    """
    Takes the given string in the format <layer_def>:<layer_def>:...:<layer_def> and parses it to produce valid
    architecture definition
    :param arch_str: The architecture definition string
    :param func_dict: A dictionary of functions
    :return: A tuple consisting of:
            1. list of layer dictionaries.
            2. input vector size
            3. output vector size
    """

    arch_list = []

    for ldef in arch_str.split(":"):
        layer = dict(  )
        for lprop in ldef.split(","):
            if re.match('^[0-9]+$', lprop) is not None:
                layer['size'] = int(lprop)
            elif lprop in func_dict:
                layer['func'] = func_dict[lprop]
            elif re.match('^d(ense)?$', lprop) is not None:
                layer['type'] = "dense"
            elif re.match('^c(onvolution)?$', lprop) is not None:
                layer['type'] = "conv"
            elif re.match('^c(onvolution)?3d?$', lprop) is not None:
                layer['type'] = "conv3d"
            elif re.match('^p(ool)?$', lprop) is not None:
                layer['type'] = "pool"
            elif lprop in ("valid", "same"):
                layer['padding'] = lprop
            elif re.match('^[0-9]+(x[0-9]+)*$', lprop) is not None:
                shape = [int(dim) for dim in lprop.split("x")]
                if not arch_list:
                    layer['sample_shape'] = shape
                elif len(shape) == 2: # This should be a stride
                    layer['strides'] = shape
                else: # This is the filter definition
                    layer['filters'] = shape[-1]
                    layer['filter_shape'] = shape[:-1]
            else:
                raise ValueError(ldef, "Invalid layer definition!")

        arch_list.append(layer)

    return arch_list


def nn_layer_size(layer):
    """
    Finds the size of the layer, no matter what type of shape it has.
    :param layer: The standard definition of a layer in dictionary as returned from `nn_parse_architecture`
    :return: The number of input values in each sample.
    """
    if 'sample_shape' in layer:
        sample_shape = layer['sample_shape']
        return np.prod(sample_shape)
    elif 'size' in layer:
        return layer['size']
    else:
        return None


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def maybe_download(filename, workdir, source_url):
    """Download the data from Yann's website, unless it's already here."""

    if not os.path.exists(workdir):
        os.mkdir(workdir)
    filepath = os.path.join(workdir, filename)
    if not os.path.exists(filepath):
        eprint('Downloading %s' % filename)

        filepath, _ = urlretrieve(source_url + filename, filepath)
        statinfo = os.stat(filepath)
        eprint('Successfully downloaded %s (%d bytes)' % (filename, statinfo.st_size))
    return filepath


def packed_images_reader(filename, workdir='MNIST_data',
                        source_url="http://yann.lecun.com/exdb/mnist/",
                        normalize=True):
    path = maybe_download(filename, workdir, source_url)

    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    eprint('Extracting %s' % path)
    with gzip.open(path) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in packed image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if normalize:
            data = np.multiply(data, 1.0 / 255.0)
        return data, num_images, rows, cols


if __name__ == "__main__":
    funcs = { 'relu': np.multiply, 'elu': np.prod }

    arch = nn_parse_architecture("28x28x1:c,5x5x32,relu,valid,2x2:c,5x5x16,relu,VALID,2x2:50", funcs)
    assert len(arch) == 4
    assert arch[0]['sample_shape'] == [28, 28, 1]
    assert arch[2]['padding'] == "valid"
    assert arch[1]['strides'] == [2, 2]
    assert nn_layer_size(arch[0]) == 784

    arch = nn_parse_architecture("6957:d,2000,elu:d,1000,elu:150", funcs)
    assert len(arch) == 4
    assert arch[0]['size'] == 6957
    assert arch[1]['size'] == 2000
    assert arch[1]['func'] == np.prod
    assert nn_layer_size(arch[0]) == 6957

    print("Success!")
