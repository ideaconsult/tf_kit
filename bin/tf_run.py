#!/usr/bin/env python

#####
# Tensor flow model training and testing tool. Can work with different models.
# File: The command line tool for invoking the AE class.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev

from tf_kit.tf_hooks import *
from tf_kit.tf_persist import *
from tf_kit.dt_utils import *
import sys
import argparse


def _prepare_test_data(data, format):
    if format == "images":
        img_data, num, w, h = packed_images_reader(data)
        return np.reshape(img_data, newshape=[num, w * h])
    else:
        return data


# Deal with the arguments.
argp = argparse.ArgumentParser(
    description="A TensorFlow model running tool for different models and architectures."
)

argp.add_argument('-t', '--test-set', type=str, dest="test", nargs='+', required=True,
                  help="The list of data files to be used for training.")
argp.add_argument('-d', '--delimiter', type=str, required=False, metavar="delimiter", default='\t',
                  dest="delimiter", help="The delimiter to be expected in the data files. Default is tab.")
argp.add_argument('-o', '--output', type=str, dest="output", nargs="*",
                  help="The output file to write the results to. Default is none.")
argp.add_argument('-m', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
                  help="The path to the model - both for retrieving and storing.")
argp.add_argument('-c', '--checkpoint', type=int, dest="checkpoint", default=None,
                  help="Checkpoint number to be used for model restoring.")

argp.add_argument('-g', '--generate', required=False, action="store_true", dest="generate",
                  help="Whether to run the model in generative mode, feeding the latent and printing the output."
                       " The normal mode is `inference` - feed the input and print the latent.")

argp.add_argument('-q', '--quiet', required=False, action="store_true", dest="quite",
                  help="Whether to suppress the more detailed info messages.")

args = argp.parse_args()

if not args.quite:
    tf.logging.set_verbosity(tf.logging.INFO)

# Restore the model
model, saver, path, checkpoint = tf_restore_graph(args.model_path)

run_fn = tf_generative_run if args.generate else tf_inference_run

assert model is not None
assert saver is not None

tf_logging.info("Running a model retrieved from: %s" % path)
display_step = 10

with tf.train.MonitoredSession(tf.train.ChiefSessionCreator(scaffold=tf.train.Scaffold(saver=saver),
                                                            checkpoint_dir=os.path.dirname(path))) as sess:
    i = 0
    for inf in args.test:
        tf_logging.info("Acting on: %s" % inf)
        input = dt_prepare_iterator(args.test,
                                    data_format=DT_FORMAT_FILE,
                                    delimiter=args.delimiter,
                                    batch_size=model.batch_size,
                                    allow_smaller_batch=True,
                                    num_epochs=1)
        if args.output is None:
            run_fn(sess, model, iterator=input,
                   output_stream=sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout)
        else:
            out_name = args.output[i]
            tf_logging.info("Output to: %s" % out_name)

            with open(out_name, mode="wb") as outf:
                run_fn(sess, model, iterator=input, output_stream=outf)
