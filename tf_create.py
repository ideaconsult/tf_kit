#!/usr/bin/env python
#
# A generic, model creating script, capable of building different models, based on the
# command line parametes. Currently supported: vae, ae
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev
#

import tensorflow as tf
import argparse
import os
from tf_kit import *
from tf_kit.nn_utils import *
from tf_kit.tf_utils import *
from tf_kit.dt_constants import *
from tf_kit.tf_persist import tf_export_graph

argp = argparse.ArgumentParser(
    description="A tool for creating models in TensorFlow, without running _any_ training, or even "
                "initializing a session. Use `tf_train` for actual training."
)

model_list = [m for m in TF_MODELS.keys()]
argp.add_argument('model', type=str, help="The type of the model to be created. Supported: {}.".format(model_list))

argp.add_argument('arch_description', type=str,
                  help="The description of the architecture, consisting of layer definitions separated by semicolon (:).")

argp.add_argument('-p', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
                  help="The path to the model - both for retrieving and storing.")
argp.add_argument('-s', '--scope', type=str, required=False, default=None,
                  help="The model-wise scope to put all tensors into. Defaults to None.")
argp.add_argument('-b', '--batch', type=int, required=False, metavar="batch_size", default=DEF_BATCH, dest="batch_size",
                  help="The size of the mini-batches. Default is %d." % DEF_BATCH)
argp.add_argument('-c', '--cost', type=str, required=False, metavar="cost_function", default=DEF_COST_FUNC, dest="cost_function",
                  help="The cost function for be used for reconstruction part: xentropy|mse. "
                       "Default is `%s`." % DEF_COST_FUNC)
argp.add_argument('-l', '--learning-rate', type=float, required=False, metavar="learning_rate",
                  default=DEF_LEARNING_RATE,
                  dest="learning_rate", help="The AdamOptimizaer learning rate. Default is %.5f" % DEF_LEARNING_RATE)
argp.add_argument('-r', '--order', type=str, required=False, metavar="order", default=DEF_DATA_FORMAT, dest="data_format",
                  help="The order of input data, if it is not a flat vector: `hwc` or `chw`. "
                       "Default is `%s`." % DEF_DATA_FORMAT)
argp.add_argument('-n', '--noise', type=float, required=False, default=.0,
                  help="Applicable for AEs only - The amount of noise to induce in the latents - variance. "
                       "Default is 0.")
argp.add_argument('-w', '--equal-weights', required=False, action="store_true", dest="equal_weights",
                  help="Applicable for AEs only - whether to force symmetrical weights "
                       "in the encoder and decoder be the same.")

argp.add_argument('-q', '--quiet', required=False, action="store_true", dest="quite",
                  help="Whether to suppress the more detailed info messages.")

args = argp.parse_args()
if not args.quite:
    tf.logging.set_verbosity(tf.logging.INFO)

arch = nn_parse_architecture(args.arch_description, TF_FUNC_DICT)
if args.model not in TF_MODELS:
    raise NotImplementedError("Unsupported model type: %s" % args.model)

Model = TF_MODELS[args.model]

if args.scope is None:
    Model(arch, **vars(args))
else:
    tf.logging.info("Scoping the model in `%s`" % args.scope)
    with tf.variable_scope(args.scope):
        Model(arch, **vars(args))

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)

tf_export_graph(args.model_path, use_meta_graph=True)
