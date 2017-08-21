#
# General TensorFlow made-easy tools, primary focused on architecture building.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev
"""
The TensorFlow persistent provides this, placeholder class:

@@RestoredModel
@@WrappedModel

"""

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from .tf_utils import tf_tensors_size
import os

META_GRAPH_BASE = "model.ckpt"
GRAPH_FILENAME = "graph.pb"
META_EXT = ".meta"


class RestoredModel:
    def __init__(self, new_input=None, collection_idx=0):
        """
        Make a fake model out of already restored collections in the current graph
        :param new_input: The new input to be saved, rather than the saved in the collection
        :param collection_idx: The index in the collection, from which to retrieve tensors.
        """
        self.input_var = tf.get_collection("inputs")[collection_idx] if new_input is None else new_input
        self.output_var = tf.get_collection("outputs")[collection_idx]

        latent_col = tf.get_collection("latents")
        self.latent_var = latent_col[collection_idx] if collection_idx < len(latent_col) else None

        gen_col = tf.get_collection("generators")
        self.generator_var = gen_col[collection_idx] if collection_idx < len(gen_col) else None

        target_col = tf.get_collection("targets")
        self.target_var = target_col[collection_idx] if collection_idx < len(target_col) else None

        self.loss_op = tf.get_collection("losses")[collection_idx]
        self.train_op = tf.get_collection("train_ops")[collection_idx]

        self.output_size = int(self.output_var.get_shape()[1])

        in_shape = self.input_var.get_shape()
        self.batch_size = int(in_shape[0])
        self.input_size = int(in_shape[1])


class WrappedModel:
    def __init__(self, x_in, y_out, loss_op, train_op, y_target=None, latent_in=None, generative_out=None):
        """
        Make a fake model out of already created ops and variables and just provide the
        unified access to it. Also, adds these to the appropriate collections.
        :param x_in: The input tensor, if present.
        :param y_out: The output tensor.
        :param loss_op: The loss calculating op.
        :param train_op: The training op.
        :param y_target: The target tensor, if present.
        :param latent_in: The input of latents for generative models.
        :param generative_out: The output of generation, for generative models.
        """

        self.input_var = x_in
        self.output_var = y_out
        self.target_var = y_target
        self.loss_op = loss_op
        self.train_op = train_op

        in_shape = x_in.get_shape()
        out_shape = y_out.get_shape()
        self.batch_size = int(in_shape[0])
        self.input_size = int(in_shape[1])
        self.output_size = int(out_shape[1])

        tf.add_to_collection("inputs", x_in)
        tf.add_to_collection("outputs", y_out)
        tf.add_to_collection("targets", y_target if y_target is not None else tf.zeros([self.batch_size], dtype=tf.int32))
        self.latent_var = latent_in
        tf.add_to_collection("latents", latent_in if latent_in is not None else tf.zeros(out_shape))
        self.generator_var = generative_out
        tf.add_to_collection("generators", generative_out if generative_out is not None else tf.zeros(in_shape))

        tf.add_to_collection("train_ops", train_op)
        tf.add_to_collection("losses", loss_op)
        tf.summary.scalar("loss", loss_op)


def tf_export_graph(model_path, use_meta_graph=True):
    """
    Export the currently constructed graph into the given file. 
    :param model_path: The path to store the graph to.
    :param use_meta_graph: Whether to actually export the whole meta graph.
    """

    if use_meta_graph:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        tf.train.export_meta_graph(model_path, clear_devices=True)
    else:
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        model_path = tf.train.write_graph(graph_def,
                                          logdir=os.path.dirname(model_path),
                                          name=os.path.basename(model_path),
                                          as_text=True)

    trainables = tf.trainable_variables()
    tf_logging.info("Model created in {}\n"
                    "  ...trainable tensors = {:d}\n"
                    "  ...total parameters  = {:d}.".format(
        model_path,
        len(trainables),
        tf_tensors_size(trainables, collapse=True))
    )


def tf_restore_graph(model_path, input_pipe=None, from_scope=None):
    """
    Loads the saved graph and restores the model and the trained variables from given path.
    :param model_path: The path to the saved model to restore from.
    :param input_pipe: The new input_pipe for the model, if needed.
    :param from_scope: Search for specific ops inside this scope.
    :return: The RestoredModel instance.
    """

    checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
    if checkpoint_path is None:
        # We're dealing with graph_def
        tf_logging.info("Restoring from graph from `%s`..." % model_path)
    else:
        # TODO: Take into account the passed checkpoint number here, modifying checkpoint_path!
        model_path = checkpoint_path + META_EXT
        tf_logging.info("Restoring from checkpoint from `%s`..." % checkpoint_path)

    # Since each time we provide different input, we must initialize new saver, so it can take care for
    # the newly added variables, thus - we have different scenarios for both graph restorations.
    input_name = "input_pipe" if from_scope is None else (from_scope + ":" + "input_pipe")
    if input_pipe is None:
        # This is pure 'resume previous training' mode.
        saver = tf.train.import_meta_graph(model_path)
        input_pipe = tf.get_default_graph().get_tensor_by_name(input_name + ":0")
    else:
        # This is 'train with this, new data'
        tf.train.import_meta_graph(model_path,
                                   input_map={input_name: input_pipe},
                                   import_scope="restored")
        saver = None

    model = RestoredModel(input_pipe)
    trainables = tf.trainable_variables()
    tf_logging.info("Done restoring:\n"
                    "  ...trainable tensors = {:d}\n"
                    "  ...total parameters  = {:d}.".format(
        len(trainables),
        tf_tensors_size(trainables, collapse=True)))
    return model, saver, model_path, checkpoint_path


