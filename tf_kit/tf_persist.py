#
# General TensorFlow made-easy tools, primary focused on architecture building.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from .tf_utils import tf_tensors_size
import os

META_GRAPH_BASE = "model.ckpt"
GRAPH_FILENAME = "graph.pb"
META_EXT = ".meta"


class RestoredModel:
    def __init__(self, new_input=None):
        """
        Make a fake model out of already restored collections in the current graph
        :param new_input: The new input to be saved, rather than the saved in the collection
        """
        self.input_var = tf.get_collection("inputs")[0] if new_input is None else new_input
        self.output_var = tf.get_collection("outputs")[0]
        self.latent_var = tf.get_collection("latents")[0]
        self.generative_var = tf.get_collection("generators")[0]
        self.loss_op = tf.get_collection("losses")[0]
        self.train_op = tf.get_collection("train_ops")[0]

        self.batch_size = int(self.latent_var.get_shape()[0])


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


def tf_restore_graph(model_path, input_pipe=None):
    checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
    if checkpoint_path is None:
        # We're dealing with graph_def
        tf_logging.info("Restoring from initial graph...")
    else:
        # TODO: Take into account the passed checkpoint number here, modifying checkpoint_path!
        model_path = checkpoint_path + META_EXT
        tf_logging.info("Restoring from checkpoint...")

    # Since each time we provide different input, we must initialize new saver, so it can take care for
    # the newly added variables, thus - we have different scenarios for both graph restorations.
    if input_pipe is None:
        # This is pure 'resume previous training' mode.
        saver = tf.train.import_meta_graph(model_path)
        input_pipe = tf.get_default_graph().get_tensor_by_name("input_pipe:0")
    else:
        # This is 'train with this, new data'
        tf.train.import_meta_graph(model_path,
                                   input_map={'input_pipe': input_pipe},
                                   import_scope="restored")
        saver = None

    model = RestoredModel(input_pipe)
    tf_logging.info("... done.")
    return model, saver, model_path, checkpoint_path


