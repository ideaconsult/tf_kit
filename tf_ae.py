#!/usr/bin/env python
#
# Auto-Encoder based dimensionality reduction
#
# The generic Autoencoder implementation, capable of building different size and type of models, based on initialization
# parameters.
#
# For example building a model that has one input, two hidden and (of course) the latent layers, fully connected,
# should be initialized like this:
#
# ae = AutoEncoder(architecture=[
#           { 'size': 6000 }, # The input vector size
#           { 'size': 2000, 'type': "dense", 'func': tf.nn.elu },
#           { 'size': 1000, 'type': "dense", 'func': tf.nn.elu },
#           { 'size': 100 }  # The latent space size
#       ],
#       batch_size=BATCH)
#
# Convolution layers are also possible. For example the MNIST set AE could be initialized like this:
#
# ae = AutoEncoder(architecture=[
#           { 'sample_shape': [28,28,1] }, # The input sample shape
#           { 'type': "conv", 'filter_shape': [5,5], 'strides': [2,2], 'filters': 16, 'padding': "valid" },
#           { 'type': "conv", 'filter_shape': [5,5], 'strides': [2,2], 'filters': 32, 'padding': "same" },
#           { 'size': 50 } # The latent space size
#       ],
#       batch_size=BATCH)
#
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev
#

from tensorflow.python.platform import tf_logging
from .tf_persist import tf_export_graph
from .tf_utils import *
from .nn_utils import *
import re

DEF_BATCH=100
DEF_LEARNING_RATE=.001
DEF_DATA_FORMAT="hwc"
DEF_COST_FUNC="xentropy"
DEF_NOISE = .0


class AutoEncoder():
    """ Auto-encoder (AE) with a dynamically initialized architecture.
    """

    def __init__(self,
                 architecture,
                 input_pipe=None,
                 learning_rate=None,
                 equal_weights=False,
                 noise_variance=DEF_NOISE,
                 batch_size=DEF_BATCH,
                 data_format=DEF_DATA_FORMAT,
                 cost_function=DEF_COST_FUNC):

        """
        Initialize and construct the Auto-encoder architecture.
        :param architecture: The architectue as returned from `nn_utils.nn_parse_architecture`.
        :param input_pipe: If provided, this will be the data used for training.
        :param learning_rate: The learning rate to be used with the AdamOptimizer.
        :param equal_weights: Whether to impose equal-weights restriction of corresponding layers
                              of the encoder and decoder.
        :param noise_variance: Whether to induce noise in the latent variables before decoding,
                               turning this into Denoising Auto-encoder.
        :param batch_size: The size of the mini-batches.
        :param data_format: The data ordering format - `cwh` or `hwc`. Default is the later.
        :param cost_function: Cost function to be used: `xentropy` (default), `mse`, `abs`, `hinge` or `cos`.
        """
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.input_size = nn_layer_size(architecture[0])
        self.latent_size = nn_layer_size(architecture[-1])
        self.architecture = architecture[1:-1]
        self.cost_function = cost_function
        self.data_format = 'N' + data_format.upper()
        self.final_func = architecture[-1]['func'] if 'func' in architecture[-1] else tf.nn.sigmoid
        self.noise_variance = noise_variance
        self.equal_weights = equal_weights

        # Initialize, and possible reshape the input
        if input_pipe is not None:
            self.x_in = input_pipe
        else:
            self.x_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.input_size], name="input_pipe")
        tf.add_to_collection("inputs", self.x_in)

        self.x_shaped = self.x_in if 'sample_shape' not in architecture[0] else \
            tf.reshape(self.x_in, shape=[-1] + architecture[0]['sample_shape'])

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_op()

    def _create_network(self):
        # Use recognition network to calculate the latent variables
        self.z_out = self._recognition_network()
        tf.add_to_collection("latents", self.z_out)

        # If we have noise_variance > 0, this means we're making a denoising auto-encoder,
        # meaning we have to induce some noise in the latent variables, before initiating the
        # reconstruction phase.
        if self.noise_variance > .0:
            eps = tf.random_normal((self.batch_size, self.latent_size),
                                   mean=.0,
                                   stddev=self.noise_variance,
                                   dtype=tf.float32)
            self.z_latent = tf.add(self.z_out, eps)
        else:
            self.z_latent = self.z_out

        tf.add_to_collection("generators", self.z_latent)

        # Use generator to reconstruct the input back
        self.x_decoded = self._generator_network()
        tf.add_to_collection("outputs", self.x_decoded)

    def _recognition_network(self):
        last_input = tf_build_architecture(self.architecture,
                                               batch_in=self.x_shaped,
                                               scope_prefix="recognize",
                                               data_format=self.data_format)

        last_input = tf_ensure_flat(last_input)

        # The hidden layers are ready - now build the latent variable one
        return tf_dense_layer("latent_z", last_input,
                                  params={ 'size': self.latent_size },
                                  empty_func=True)

    def _generator_network(self):
        final_layer = { 'func': self.final_func,
                        'size': self.input_size,
                        'type': "dense"
                        }

        # Now prepare the reuse dictionary for weights sharing
        if self.equal_weights:
            reuse_dict = dict()
            arch_len = len(self.architecture)
            for var in tf.trainable_variables():
                var_m = re.search('recognize_([0-9]+)/weights', var.name)
                if var_m is not None:
                    shape_perm = [i for i in range(var.get_shape().ndims)]
                    shape_perm[-2], shape_perm[-1] = shape_perm[-1], shape_perm[-2]
                    var_name = 'generate_%d/weights' % (arch_len - int(var_m.group(1)) + 2)
                    reuse_dict[var_name] = tf.transpose(var, perm=shape_perm)
        else:
            reuse_dict = None
        rev_arch = tf_reverse_architecture(self.architecture,
                                               final_layer=final_layer,
                                               batch_size=self.batch_size)

        # In any case we need to invoke creating the reversed architecture
        last_input = tf_build_architecture(rev_arch,
                                               batch_in=self.z_out,
                                               scope_prefix="generate",
                                               transpose=True,
                                               data_format=self.data_format,
                                               reuse_dict=reuse_dict)

        return tf_ensure_flat(last_input)

    def _create_loss_op(self):
        """
        Constructs the loss optimizer for the AE, according to the given loss function.
        """
        loss_fn = tf_loss_function(self.cost_function)
        self.loss_op = loss_fn(self.x_in, self.x_decoded)

        tf.add_to_collection("losses", self.loss_op)
        tf.summary.scalar("loss", self.loss_op)

        if self.learning_rate is not None:
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op,
                                                                                              global_step=global_step)
            tf.add_to_collection("train_ops", self.train_op)
            tf_logging.info("Added AdamOptimizer with learning rate: %.8f" % self.learning_rate)


    @property
    def input_var(self):
        return self.x_in

    @property
    def output_var(self):
        return self.x_decoded

    @property
    def latent_var(self):
        return self.z_out

    @property
    def generator_var(self):
        return self.z_latent


# Now if we're directly invoked - create the model according to command line parameters
# and store it in the given location. No training happens from this script!
if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser(
        description="A Variational Encoder model created with TensorFlow and stored in the "
                    "given location for further training / using."
    )

    argp.add_argument('architecture', type=str,
                      help="The actual architecture consisting of layer definitions separated by semicolon (:).")

    argp.add_argument('-m', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
                      help="The path to the model - both for retrieving and storing.")
    argp.add_argument('-b', '--batch', type=int, required=False, metavar="batch_size", default=DEF_BATCH, dest="batch",
                      help="The size of the mini-batches. Default is %d." % DEF_BATCH)
    argp.add_argument('-c', '--cost', type=str, required=False, metavar="cost_function", default=DEF_COST_FUNC, dest="cost",
                      help="The cost function for be used for reconstruction part: xentropy|mse. "
                           "Default is `%s`." % DEF_COST_FUNC)
    argp.add_argument('-l', '--learning-rate', type=float, required=False, metavar="learning_rate",
                      default=DEF_LEARNING_RATE,
                      dest="learning_rate", help="The AdamOptimizaer learning rate. Default is %.5f" % DEF_LEARNING_RATE)
    argp.add_argument('-r', '--order', type=str, required=False, metavar="order", default=DEF_DATA_FORMAT, dest="order",
                      help="The order of input data, if it is not a flat vector: `hwc` or `chw`. "
                           "Default is `%s`." % DEF_DATA_FORMAT)
    argp.add_argument('-n', '--noise', type=float, required=False, default=.0,
                      help="Applicable for AEs only - The amount of noise to induce in the latents - variance. "
                           "Default is 0.")
    argp.add_argument('-w', '--equal-weights', required=False, action="store_true", dest="equal_weights",
                      help="Applicable for AEs only - whether to force symmetrical weights "
                           "in the encoder and decoder be the same.")
    args = argp.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)

    arch = nn_parse_architecture(args.architecture, TF_FUNC_DICT)

    model = AutoEncoder(arch,
                        batch_size=args.batch,
                        learning_rate=args.learning_rate,
                        cost_function=args.cost,
                        data_format=args.order,
                        noise_variance=args.noise,
                        equal_weights=args.equal_weights)

    tf_export_graph(args.model_path, use_meta_graph=True)
