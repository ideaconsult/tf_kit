#
# General TensorFlow made-easy tools, primary focused on architecture building.
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev

import tensorflow as tf
import numpy as np
import re

TF_SELU_LAMBDA = 1.0507009873554804934193349852946
TF_SELU_ALPHA = 1.6732632423543772848170429916717

# The dictionary used during normal architecture
TF_FUNC_DICT = { 'relu': tf.nn.relu,
                 'relu6': tf.nn.relu6,
                 'elu': tf.nn.elu,
                 'selu': lambda x, name: tf.multiply(tf.where(x >= .0, x, tf.nn.elu(x) * TF_SELU_ALPHA),
                                                     TF_SELU_LAMBDA, name=name),
                 'sigmoid': tf.nn.sigmoid,
                 'tanh': tf.nn.tanh,
                 'id': tf.identity }

TF_FORMAT_PIPE = "tf_pipe"
TF_FORMAT_CONST = "tf_mem"


def tf_tensors_size(inputs, collapse=False):
    """
    Calculates the number of all variables involved in the given tensor, regarding of the shape
    :param inputs: A list of input tensors
    :param collapse: If True it'll sum all the tensors' sizes to one single scalar.
    :return: A list of sizes for each of the given tensors or a single scalar, if collapse=True
    """

    sizes = []
    for t in inputs:
        sizes.append(np.prod([int(x) for x in t.get_shape()]))
    return np.sum(sizes, dtype=int) if collapse else sizes


def tf_const_input(filename, delimiter=None, shape=None):
    np_data = np.loadtxt(filename, delimiter=delimiter)
    return tf.constant(np_data, dtype=tf.float32, shape=shape)


def tf_csv_reader(filename_queue, input_size, processor=None, delimiter=' '):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    if processor is None:
        cols = list(tf.decode_csv(value,
                                  record_defaults=[[.0]] * input_size,
                                  field_delim=delimiter,
                                  name="CSV_reader"))
        return tf.stack(cols)
    else:
        return processor(value, input_size)


def tf_file_pipe(filenames, num_epochs=10, reader=tf_csv_reader, shuffle=True, delimiter=None, name=None):
    """
    A batch loader for given set of filenames. Any reader can be passed, but CSV is by default
    :param filenames: The list of names to be processed
    :param num_epochs: Number of times to iterate over the files.
    :param reader: The actual file reader/parsed to be used.
    :param shuffle: Whether to shuffle file names.
    :param delimiter: The delimiter to be used in parsing the file
    :param name: The name for the data tensor.
    :return:
    """
    with open(filenames[0]) as f:
        line = f.readline()
        input_size = len(line.split(sep=delimiter))

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    return reader(filename_queue, input_size, delimiter=delimiter)


def tf_pipe_shuffle(sample, shape=None, batch_size=100, allow_smaller_final_batch=False, name=None):
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    if shape is not None:
        shape = [shape]
    return \
        tf.train.shuffle_batch([sample], shapes=shape,
                               batch_size=batch_size, capacity=capacity,
                               min_after_dequeue=min_after_dequeue,
                               allow_smaller_final_batch=allow_smaller_final_batch,
                               name=name)


def tf_prepare_pipe(file_list, data_format,
                    num_epochs=10,
                    shuffle=True,
                    delimiter=' ',
                    batch_size=100,
                    allow_smaller_batches=False,
                    name=None):
    """
    Prepares a list of data files to be fed into TF model.
    :param file_list: The list of filenames.
    :param data_format: A format of the data `tf_pipe` or `tf_mem`.
    :param num_epochs: Number of epochs to run the algorithm.
    :param shuffle: Whether to shuffle the names, or not.
    :param delimiter: Delimiter used in the file.
    :param batch_size: Required batch size for the data.
    :param allow_smaller_batches: Whether the last batch is allowed to be smaller.
    :param name: The name for the tensor which is returned.
    :return:
    """
    if data_format == TF_FORMAT_CONST:
        data = tf.train.input_producer(tf_const_input(file_list[0], delimiter=delimiter),
                                       num_epochs=num_epochs,
                                       shuffle=shuffle).dequeue()
    elif data_format == TF_FORMAT_PIPE:
        data = tf_file_pipe(file_list, num_epochs=num_epochs,
                            name=name if not shuffle else None,
                            shuffle=shuffle,
                            delimiter=delimiter)
    else:
        return None

    if shuffle:
        data = tf_pipe_shuffle(data, batch_size=batch_size,
                               allow_smaller_final_batch=allow_smaller_batches,
                               name=name)
    else:
        data = tf.train.batch([data],
                              batch_size=batch_size,
                              allow_smaller_final_batch=allow_smaller_batches,
                              name=name)
    return data


def tf_get_reuse_variable(name, shape, initializer=None, reuse_dict=None, variables_collection=None):
    """
    Tries to find a variable with `name` in the current variable scope, if not - makes a new one.
    :param name: The name of the variable to be created / reused.
    :param shape: The shape of newly created variable.
    :param initializer: The initializer to be used for the variable creation.
    :param reuse_dict: The dictionary of reusable variable to search into.
    :param variables_collection: A collection of variables to add it to.
    :return: A tensor with either a newly crated variable or reused one.
    """
    scope = tf.get_variable_scope().name
    if reuse_dict is not None and (scope + '/' + name) in reuse_dict:
        rvar = reuse_dict[scope + '/' + name]
        return tf.reshape(rvar, shape=shape) if rvar.get_shape() != shape else rvar
    else:
        var = tf.get_variable(name, shape=shape, initializer=initializer)
        if variables_collection is not None:
            tf.add_to_collection(variables_collection, var)
        return var


def tf_dense_layer(scope, x, params,
                   variables_collection=None,
                   reuse_dict=None,
                   dropout_keeprate=None):
    """
    Creates a dense layer, that have weights from all elements
    from the given input `x`, according to the parameters dictionary passed.
    The elements recognized in this dictionary are:
        size: Number of neurons in the layer
        func: The transition function that should be applied. Optional.
        initializer: The initializer to be used for weights. Defaults to Xavier.

    :param scope: The tf scope to restrict the variables into
    :param x: The input tensor. Can have it's first dimension unspecified
    :param params: The params dictionary. See above.
    :param variables_collection: A collection to add the variables to.
    :param reuse_dict: A dictionary of tensors to be reused instead of creating new ones.
    :param dropout_keeprate: If dropout is taking place - this is the tensor to be used.
    :return: A tuple (tensor resulting from the dense layer construction, list of model variables)
    """

    x = tf_ensure_flat(x)
    in_size = int(x.get_shape()[1])
    the_size = params["size"]
    func = params.get("func", tf.nn.relu)
    dropout = params.get('dropout', False)
    initializer = params.get("initializer", tf.contrib.layers.xavier_initializer(False))

    with tf.variable_scope(scope):
        weights = tf_get_reuse_variable("weights", [in_size, the_size],
                                        initializer=initializer,
                                        reuse_dict=reuse_dict,
                                        variables_collection=variables_collection)
        biases = tf_get_reuse_variable("biases", [the_size],
                                       initializer=tf.constant_initializer(.0),
                                       reuse_dict=reuse_dict,
                                       variables_collection=variables_collection)

    output = tf.add(tf.matmul(x, weights), biases, name=scope) if func is None else \
        func(tf.add(tf.matmul(x, weights), biases), name=scope)
    return tf.nn.dropout(output, keep_prob=dropout_keeprate) if dropout and dropout_keeprate is not None else output


def tf_conv_layer(scope, x, params,
                  transpose=False,
                  data_format="NHWC",
                  variables_collection=None,
                  reuse_dict=None,
                  dropout_keeprate=None):
    """
    Creates a convolution layer, operating on the given input `x` and constructed, based
    on the information from the given `params`. The dictionary can have the following
    properties:
        filter_shape: The shape/size of the filter
        filters: The number of filters/features for this layer.
        strides: The convolution stride, without the 1s at both side.
        func: The non-linearity induced after the convolution. Optional.
        padding: The striding padding - SAME or VALID. Default is VALID.
        initializer: The initializer to be used for weights. Defaults to
                tf.truncated_normal_initializer(stddev=.02)

    The function parametes are:

    :param scope: The tf scope to restrict the variables into
    :param x: The input tensor. It'll be reshaped, if `input_shape` is present in the dict.
    :param params: The parameters dictionary.
    :param transpose: Whether a transposed convolution should be constructed. Default is False.
    :param data_format: The ordering of data to be expected from the input tensor.
    :param variables_collection: A collection to add the variables to.
    :param reuse_dict: A dictionary of tensors to be reused instead of creating new ones.
    :param dropout_keeprate: If dropout is taking place - this is the tensor to be used.
    :return: A tuple (tensor resulting from the layer construction, list of model variables)
    """

    in_shape = params.get('input_shape', None)
    out_shape = params.get('output_shape', None)
    transpose = transpose or params.get('transpose', False)
    dropout = params.get('dropout', False)

    x_shape = x.get_shape()
    if in_shape is None or out_shape is None:
        in_shape = x_shape
        filters = params["filters"]
    else:
        if out_shape[0] is None:
            out_shape[0] = -1
        filters = int(out_shape[-1]) if data_format == "NHWC" else int(out_shape[1])

    # We might need to reshape, no matter in what direction we're working
    if not x_shape.ndims == len(in_shape):
        if in_shape[0] is None:
            in_shape[0] = -1
        x = tf.reshape(x, shape=in_shape)
        params['input_shape'] = in_shape

    in_size = int(in_shape[-1]) if data_format == "NHWC" else int(in_shape[1])
    strides = [1] + params["strides"][:] + [1]
    w_shape = params["filter_shape"] + ([in_size, filters] if not transpose else [filters, in_size])
    func = params.get("func", tf.nn.relu)
    padding = params.get("padding", "valid").upper()
    initializer = params.get("initializer", tf.truncated_normal_initializer(stddev=.02))

    with tf.variable_scope(scope):
        weights = tf_get_reuse_variable("weights", shape=w_shape,
                                        initializer=initializer,
                                        reuse_dict=reuse_dict,
                                        variables_collection=variables_collection)
        biases = tf_get_reuse_variable("biases", shape=[filters],
                                       initializer=tf.constant_initializer(.0),
                                       reuse_dict=reuse_dict,
                                       variables_collection=variables_collection)

        conv = tf.nn.conv2d(x, weights,
                            strides=strides,
                            padding=padding,
                            data_format=data_format) if not transpose else \
            tf.nn.conv2d_transpose(x, weights,
                                   strides=strides,
                                   output_shape=out_shape,
                                   padding=padding,
                                   data_format=data_format)

    output = tf.add(conv, biases, name=scope) if func is None else func(tf.add(conv, biases), name=scope)
    return tf.nn.dropout(output, keep_prob=dropout_keeprate) if dropout and dropout_keeprate is not None else output


def tf_build_architecture(architecture,
                          batch_in,
                          scope_prefix,
                          transpose=False,
                          data_format="NHWC",
                          variables_collection=None,
                          reuse_dict=None,
                          dropout_keeprate=None):
    """
    Build the given architecture, invoking the appropriate of the above functions.
    :param architecture: The list of dictionaries describing each layer. The `type` determines which of the above
                         will be invoked.
    :param batch_in: The input for the network.
    :param scope_prefix: The prefix for the variables scope which will be used.
    :param transpose: Whether to transpose the convolution layers.
    :param data_format: The ordering of data to be expected from the input tensor.
    :param variables_collection: A collection to add variables to.
    :param reuse_dict: A dictionary of tensors to be reused instead of creating new ones.
    :param dropout_keeprate: If dropout is taking place - this is the tensor to be used.
    :return: A tuple (tensor resulting from the architecture construction, list of model variables)
    """
    last_input = batch_in
    arch_len = len(architecture)

    for idx, params in enumerate(architecture, start=1):
        if 'name' in params:
            scope = params['name']
        elif transpose:
            scope = scope_prefix + "_%d" % (arch_len - idx + 1)
        else:
            scope = scope_prefix + "_%d" % idx

        if not 'input_shape' in params:
            params["input_shape"] = last_input.get_shape().as_list()

        ltype = params["type"]
        if ltype == "dense":
            last_input = tf_dense_layer(scope,
                                        last_input,
                                        params,
                                        variables_collection=variables_collection,
                                        reuse_dict=reuse_dict,
                                        dropout_keeprate=dropout_keeprate)
        elif ltype == "conv":
            last_input = tf_conv_layer(scope,
                                       last_input,
                                       params,
                                       reuse_dict=reuse_dict,
                                       transpose=transpose,
                                       variables_collection=variables_collection,
                                       dropout_keeprate=dropout_keeprate,
                                       data_format=data_format)
        else:
            assert False

        if not 'output_shape' in params:
            params["output_shape"] = last_input.get_shape().as_list()

    return last_input


def tf_build_reverse_params(layer, params):
    ptype = params['type']
    ltype = layer['type']
    if 'func' in params:
        layer['func'] = params['func']

    if ltype == "dense":
        layer['size'] = params['size'] if ptype == "dense" else \
            np.prod(params['output_shape'][1:])
    elif layer['type'] == "conv":
        layer['input_shape'], layer['output_shape'] = layer['output_shape'], layer['input_shape']

    return layer


def tf_build_reuse_dict(vars, transpose=True):
    """
    Build a dictionary which can be used in tf_build_architecture() when
    reusing a variables is required.
    :param vars: The set of variables to be searched within.
    :param transpose: If the variables need to be transposed
    :return: A dictionary between a variable name and the tensor to be reused.
    """
    reuse_dict = dict()
    for var in vars:
        var_m = re.search('recognize_([0-9]+)/(weights)', var.name)
        if var_m is not None:
            var_name = 'generate_%d/%s' % (int(var_m.group(1)), var_m.group(2))
            if transpose:
                shape_perm = [i for i in range(var.get_shape().ndims)]
                shape_perm[-2], shape_perm[-1] = shape_perm[-1], shape_perm[-2]
                reuse_dict[var_name] = tf.transpose(var, perm=shape_perm)
            else:
                reuse_dict[var_name] = var
    return reuse_dict


def tf_reverse_architecture(architecture, final_layer, batch_size=None):
    """
    Given an architecture specification, it produces a symetrically reversed one.
    The process is not trivial reverse of given list, because dense-to-convolution
    and convolution-to-dense transitions need to handled, as well as activation functions,
    which actually shift on the next (reversed) layer. This method is used for autoencoders
    architecture building.

    :param architecture: The forward architecture to be reversed.
    :param final_layer: The final layer to put.
    :param batch_size: The batch size, which is part of some models.
    :return: A list of layers, representing the reversed architecture.
    """
    rev_arch = [tf_build_reverse_params({'type': "dense"}, architecture[-1])]

    for i in range(len(architecture) - 1, -1, -1):
        params = architecture[i]
        layer = tf_build_reverse_params(params, architecture[i - 1] if i > 0 else final_layer)

        if 'input_shape' in layer and batch_size is not None:
            layer['input_shape'][0] = batch_size
        if 'output_shape' in layer and batch_size is not None:
            layer['output_shape'][0] = batch_size

        rev_arch.append(layer)

    return rev_arch


def tf_ensure_flat(x, ndims=2):
    """
    Makes sure a tensor is flattened ONLY IF NEEDED, i.e. it has more dimensions than required ones.
    :param x: A tensor to be flattenned
    :param ndims: The number of dimensions to be left in the output tensor.
    :return: The flatened tensor, or the original, if it is fine
    """
    x_shape = x.get_shape()
    if x_shape.ndims <= ndims:
        return x

    dim_sz = 1
    for dim in x_shape[ndims - 1:]:
        dim_sz *= int(dim)

    new_shape = x_shape[:ndims - 1].concatenate([dim_sz])
    return tf.reshape(x, shape=new_shape)


def tf_uninitialized_variables(return_op=False):
    """
    Returns a list of variables that are not, yet initialized. Can be directly passed
    to tf.initialize_variables() to obtain the initialized op.
    :param return_op: If passed it'll return the initialized op, or None, if the list is empty.
    :return: Either a list of uninitialized variables (can be empty!) or directly the
            op for initializing them.
    """

    uninitialized_list = [v for v in tf.all_variables() if not tf.is_variable_initialized(v)]
    if not return_op:
        return uninitialized_list
    elif len(uninitialized_list) > 0:
        return tf.variables_initializer(uninitialized_list)
    else:
        return None


def tf_loss_function(name):
    """
    Returns the loss function from TF, based on the given name. Also, taking into account the
    different locations of loss function for different TF versions.
    :param name: The name of the loss function: "abs", "mse", "xentropy", "hinge" or "cosine".
    :return: The function object.
    :raises: ValueError exception if the provided name is unknown.
    """
    loss_base = tf.losses if hasattr(tf, 'losses') else tf.contrib.losses

    if name == 'abs':
        return loss_base.absolute_difference
    if name in ('mse', 'l2'):
        return loss_base.mean_squared_error
    elif name in ('xentropy', 'log'):
        return loss_base.log_loss
    elif name == 'softmax':
        return loss_base.softmax_cross_entropy
    elif name == "hinge":
        return loss_base.hinge_loss
    elif name in ('cosine', 'cos'):
        return loss_base.cosine_distance
    else:
        raise ValueError(name, "Unknown cost function name!")


def tf_static_iteration(sess, iterator, ops, input_op, batch_size, feeds=None, result_idx=None, iter_fn=None):
    def _make_step(x):
        if x.shape[0] < batch_size:
            padding = batch_size - x.shape[0]
            x = np.append(x, np.array([[.0] * x.shape[1]] * padding), axis=0)
        else:
            padding = 0

        feeds[input_op] = x
        result = sess.run(ops, feed_dict=feeds)

        if result_idx is not None and padding > 0:
            if isinstance(ops, tuple) or isinstance(ops, list):
                result[result_idx] = result[result_idx][:batch_size - padding]
            else:
                result = result[:batch_size - padding]
        iter_fn(result, x)

    if feeds is None:
        feeds = dict()

    if isinstance(iterator, (list, np.ndarray)):
        _make_step(iterator)
    else:
        for xx in iterator:
            if xx.shape[0] == 0:
                break
            else:
                _make_step(xx)


def tf_validation_run(sess, model, iterator):
    """

    :param sess:
    :param model:
    :param iterator:
    :return:
    """
    loss_stat = { 'loss': .0, 'count': 0 }

    def _accumulate(stat, res, x):
        if x.shape[0] == model.batch_size:
            stat["loss"] += res
            stat["count"] += 1

    tf_static_iteration(sess, iterator,
                        ops=model.loss_op,
                        input_op=model.input_var,
                        batch_size=model.batch_size,
                        result_idx=None,
                        iter_fn=lambda res, x:_accumulate(loss_stat, res, x))

    return loss_stat["loss"] / loss_stat["count"]
