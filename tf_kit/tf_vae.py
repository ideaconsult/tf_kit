#
# Variational Auto-Encoder based dimensionality reduction
#
# The generic VAE implementation, capable of building different size and type of models, based on initialization
# parameters.
#
# For example building a model that has one input, two hidden and (of course) the latent layers, fully connected,
# should be initialized like this:
#
# vae = VarAutoEncoder(architecture=[
#           { 'size': 6000 }, # The input vector size
#           { 'size': 2000, 'type': "dense", 'func': tf.nn.elu },
#           { 'size': 1000, 'type': "dense", 'func': tf.nn.elu },
#           { 'size': 100 }  # The latent space size
#       ],
#       batch_size=BATCH)
#
# Convolution layers are also possible. For example the MNIST set VAE could be initialized like this:
#
# vae = VarAutoEncoder(architecture=[
#           { 'sample_shape': [28,28,1] }, # The input sample shape
#           { 'type': "conv", 'filter_shape': [5,5], 'strides': [2,2], 'filters': 16, 'padding': "valid" },
#           { 'type': "conv", 'filter_shape': [5,5], 'strides': [2,2], 'filters': 32, 'padding': "same" },
#           { 'size': 50 } # The latent space size
#       ],
#       batch_size=BATCH)
#
# The following articles were used as reference and guidance:
#   https://jmetzen.github.io/2015-11-27/vae.html
#   http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html
#   http://kvfrans.com/variational-autoencoders-explained/
#   https://arxiv.org/abs/1312.6114
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev
#

from tensorflow.python.platform import tf_logging
from .tf_utils import *
from .nn_utils import *
from . import TF_MODELS

DEF_BATCH = 100
DEF_LEARNING_RATE = .001
DEF_DATA_FORMAT = "hwc"
DEF_COST_FUNC = "xentropy"


class VarAutoEncoder():
    """ Variation Auto-encoder (VAE) with a dynamically initialized architecture.
    """

    def __init__(self,
                 architecture,
                 input_pipe=None,
                 learning_rate=None,
                 batch_size=DEF_BATCH,
                 data_format=DEF_DATA_FORMAT,
                 cost_function=DEF_COST_FUNC,
                 **kwargs):

        """
        Initialize and construct the VAE architecture.
        :param architecture: The architecture as returned from `nn_utils.nn_parse_architecture`.
        :param input_pipe: If provided, this will be the data used for training.
        :param learning_rate: If specified, the learning rate to be used with AdamOptimizer, which will be added.
        :param batch_size: Size of mini-batches to be used.
        :param data_format: The data ordering format - `cwh` or `hwc`. Default is the later.
        :param cost_function: Cost function to be used: `xentropy` (default) or `mse`.
        """

        # self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.input_size = nn_layer_size(architecture[0])
        self.latent_size = nn_layer_size(architecture[-1])
        self.architecture = architecture[1:-1]
        self.cost_function = cost_function
        self.data_format = 'N' + data_format.upper()
        self.final_func = architecture[-1]['func'] if 'func' in architecture[-1] else tf.nn.sigmoid

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
        """
        Use recognition network to determine mean and (log) variance of 
        Gaussian distribution in latent space.
        """
        self.z_mean, self.z_log_sigma_sq = self._recognition_network()
        tf.add_to_collection("latents", self.z_mean)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.latent_size), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z_latent = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        tf.add_to_collection("generators", self.z_latent)

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_decoded = self._generator_network()
        tf.add_to_collection("outputs", self.x_decoded)

    def _recognition_network(self):
        """
        Constructs the recognition (encoder) part of the network
        :return: A two-tensor tuple with `z_mean` and `z_log_sigma_sq`
        """
        last_input = tf_build_architecture(self.architecture,
                                               batch_in=self.x_shaped,
                                               scope_prefix="recognize",
                                               data_format=self.data_format)

        last_input = tf_ensure_flat(last_input)

        # The hidden layers are ready - now build the last two, first reshaping the last layer, if needed
        z_mean = tf_dense_layer("latent_mean", last_input,
                                    params={ 'size':self.latent_size },
                                    empty_func=True)

        z_log_sigma_sq = tf_dense_layer("latent_log_sigma_sq", last_input,
                                            params= { 'size': self.latent_size },
                                            empty_func=True)
        return z_mean, z_log_sigma_sq

    def _generator_network(self):
        """
        Constructs the generator part of the network - the decoder.
        :return: A decoded input tensor.
        """
        final_layer = { 'func': self.final_func,
                        'size': self.input_size,
                        'type': "dense"
                        }

        rev_arch = tf_reverse_architecture(self.architecture, final_layer=final_layer, batch_size=self.batch_size)

        # In any case we need to invoke creating the reversed architecture
        last_input = tf_build_architecture(rev_arch,
                                           batch_in=self.z_latent,
                                           scope_prefix="generate",
                                           transpose=True,
                                           data_format=self.data_format)

        return tf_ensure_flat(last_input)

    def _create_loss_op(self):
        """
        Construct the the terms loss optimizer - `reconstruction` and `latent` losses.
        """
        # 1.) The reconstruction loss, which forces the NN towards reconstructing more accurately the
        # given input. This function is configurable, but usually it is the Bernoulli negative log-likelihood.
        if self.cost_function == 'abs':
            reconstr_loss = tf.reduce_sum(tf.abs(self.x_decoded - self.x_in), 1)
        elif self.cost_function in ('mse', 'l2', 'square'):
            reconstr_loss = tf.reduce_sum(tf.squared_difference(self.x_in, self.x_decoded), 1)
        elif self.cost_function in ('xentropy', 'log'):
            reconstr_loss = \
                -tf.reduce_sum(self.x_in * tf.log(1e-10 + self.x_decoded)
                               + (1 - self.x_in) * tf.log(1e-10 + 1 - self.x_decoded),
                               1)
        else:
            raise ValueError(self.cost_function, "Unknown cost function name!")

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1. + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

        self.loss_op = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        tf.add_to_collection("losses", self.loss_op)

        if self.learning_rate is not None:
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op,
                                                                                              global_step=global_step)
            tf.add_to_collection("train_ops", self.train_op)
            tf_logging.info("Added AdamOptimizer with learning rate: %.8f" % self.learning_rate)

        tf.summary.scalar("latent_loss", tf.reduce_mean(latent_loss))
        tf.summary.scalar("reconstruction_loss", tf.reduce_mean(reconstr_loss))
        tf.summary.scalar("loss", self.loss_op)

    @property
    def input_var(self):
        return self.x_in

    @property
    def output_var(self):
        return self.x_decoded

    @property
    def latent_var(self):
        return self.z_mean

    @property
    def generator_var(self):
        return self.z_latent


TF_MODELS['vae'] = VarAutoEncoder
