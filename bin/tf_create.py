#!/usr/bin/env python
#
# A generic, model creating script, capable of building different models, based on the
# command line parametes. Currently supported: vae, ae
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev
#

import tensorflow as tf
from tf_kit import *

DEF_BATCH=100
DEF_LEARNING_RATE=.001
DEF_DATA_FORMAT="hwc"
DEF_COST_FUNC="xentropy"
DEF_NOISE = .0

# Now if we're directly invoked - create the model according to command line parameters
# and store it in the given location. No training happens from this script!
if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser(
        description="A Variational Encoder model created with TensorFlow and stored in the "
                    "given location for further training / using."
    )

    argp.add_argument('model', type=str, help="The type of the model to be created. Either: `vae` or `ae`.")

    argp.add_argument('architecture', type=str,
                      help="The actual architecture consisting of layer definitions separated by semicolon (:).")

    argp.add_argument('-p', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
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

    if args.model in ("VAE", "vae"):
        VarAutoEncoder(arch,
                       batch_size=args.batch,
                       learning_rate=args.learning_rate,
                       cost_function=args.cost,
                       data_format=args.order)
    elif args.model in ("AE", "ae"):
        AutoEncoder(arch,
                    batch_size=args.batch,
                    learning_rate=args.learning_rate,
                    cost_function=args.cost,
                    data_format=args.order,
                    noise_variance=args.noise,
                    equal_weights=args.equal_weights)
    else:
        raise NotImplementedError("Unsupported model type: %s" % args.model)

    tf_export_graph(args.model_path, use_meta_graph=True)
