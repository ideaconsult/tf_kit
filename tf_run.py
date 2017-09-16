#!/usr/bin/env python

#
# Tensor flow model training and testing tool. Can work with different models.
# File: The command line tool for invoking the AE class.
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev

import tensorflow as tf
import numpy as np
from tf_kit.tf_hooks import *
from tf_kit.tf_persist import *
from tf_kit.dt_utils import *
from tf_kit.tf_utils import *
import sys
import os
import argparse


# Deal with the arguments.
argp = argparse.ArgumentParser(
    description="A TensorFlow model running tool for different models and architectures."
)

argp.add_argument('-t', '--test-set', type=str, dest="test", nargs='+', required=True,
                  help="The list of data files to be used for training.")
argp.add_argument('-d', '--delimiter', type=str, required=False, metavar="delimiter", default=None,
                  dest="delimiter", help="The delimiter to be expected in the data files. By default deduced from file extension.")
argp.add_argument('-o', '--output', type=str, dest="output", nargs="*",
                  help="The output file to write the results to. Default is none.")
argp.add_argument('--header', type=str, dest="header", required=False, default=None,
                  help="The header to put ont first line of each output file. Default is none.")
argp.add_argument('-f', '--output-format', type=str, dest="output_format", required=False, default="real",
                  help="The format of output data - real | int | argmax | id_argmax. Default is `real`.")
argp.add_argument('-m', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
                  help="The path to the model - both for retrieving and storing.")
argp.add_argument('-c', '--checkpoint', type=int, dest="checkpoint", default=None,
                  help="Checkpoint number to be used for model restoring.")
argp.add_argument('-i', '--model-index', type=int, dest="model_index", default=0,
                  help="The index in the model collection to retrieve it from. Default is 0.")
argp.add_argument('-g', '--generate', required=False, action="store_true", dest="generate",
                  help="Whether to run the model in generative mode, feeding the latent and printing the output."
                       " The normal mode is `inference` - feed the input and print the latent.")

argp.add_argument('-q', '--quiet', required=False, action="store_true", dest="quite",
                  help="Whether to suppress the more detailed info messages.")

args = argp.parse_args()

if not args.quite:
    tf_logging.set_verbosity(tf.logging.INFO)

# Restore the model
model, saver, path, checkpoint = tf_restore_graph(args.model_path,
                                                  checkpoint_idx=args.checkpoint,
                                                  model_index=args.model_index)

assert model is not None
assert saver is not None
assert checkpoint is not None

tf_logging.info("Running a model retrieved from: %s" % checkpoint)
display_step = 10
output_delimiter = args.delimiter

if args.output is None:
    output_stream = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout.buffer
else:
    output_stream = None

if args.output_format == "id_argmax":
    result_idx = 1

    def output_fn(res, x):
        global result_idx
        res_len = len(res)
        out_res = np.zeros([res_len, 2], dtype=np.int16)
        out_res[:, 0] = np.arange(result_idx, result_idx + res_len)
        out_res[:, 1] = np.argmax(res, axis=1)
        result_idx += res_len
        np.savetxt(output_stream, out_res, fmt="%d", delimiter=output_delimiter)
elif args.output_format == "argmax":
    def output_fn(res, x): np.savetxt(output_stream, np.argmax(res, axis=1), fmt="%d", delimiter=output_delimiter)
else:
    fmt_str = "%d" if args.output_format == "int" else "%.8f"

    def output_fn(res, x): np.savetxt(output_stream, res, fmt=fmt_str, delimiter=output_delimiter)

if args.generate:
    step_op = model.generator_out
    feed_op = model.latent_var
else:
    step_op = model.output_var
    feed_op = model.input_var

dropout_rate = tf.get_default_graph().get_tensor_by_name("dropout_rate:0")
aux_feeds = { dropout_rate: 1.} if dropout_rate is not None else None

with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    i = 0
    for inf in args.test:
        tf_logging.info("Acting on: %s" % inf)
        input = dt_prepare_iterator(inf,
                                    delimiter=args.delimiter,
                                    skip_rows=1 if args.header is not None else 0,
                                    batch_size=model.batch_size,
                                    allow_smaller_batch=True,
                                    num_epochs=1)
        if args.output is None:
            tf_static_iteration(sess, input,
                                ops=step_op,
                                input_op=feed_op,
                                batch_size=model.batch_size,
                                feeds=aux_feeds,
                                result_idx=0,
                                iter_fn=output_fn)
        else:
            out_name = args.output[i]
            tf_logging.info("Output to: %s" % out_name)
            if output_delimiter is None:
                _, f_ext = os.path.splitext(out_name)
                output_delimiter = dt_delimiter_from_ext(f_ext)

            with open(out_name, mode="wb") as outf:
                if args.header is not None:
                    bb = (str(args.header) + "\n").encode()
                    outf.write(bb)
                output_stream = outf
                tf_static_iteration(sess, input,
                                    ops=step_op,
                                    input_op=feed_op,
                                    batch_size=model.batch_size,
                                    feeds=aux_feeds,
                                    result_idx=0,
                                    iter_fn=output_fn)
