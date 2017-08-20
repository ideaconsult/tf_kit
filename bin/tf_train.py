#!/usr/bin/env python
#
# TensorFlow training tool - can take any checkpointed/saved model
# and start the training process.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev
#

import tensorflow as tf
from tf_kit.tf_utils import *
from tf_kit.tf_persist import *
from tf_kit.tf_hooks import *
from tf_kit.dt_utils import *
from tf_kit.dt_constants import *
import argparse


# Setup the command line arguments base
argp = argparse.ArgumentParser(
    description="A generic training tool, taking a built model and performing the training procedure on it.",
    epilog="If that training format is `tf_pipe` the model graph is altered with training set and epoch, "
           "becoming part of it, so future changes to it are not possible. Other methods use `feed_dict` "
           "mechanism to feed the data."
)

argp.add_argument('-t', '--training-set', type=str, dest="train", nargs='*',
                  help="The list of data files to be used for training.")
argp.add_argument('-f', '--format', type=str, required=False,
                  metavar="data_format", default=DT_FORMAT_FILE, dest="format",
                  help="The format of the input data: `images`, `file`, `array`, `tf_mem` or `tf_pipe`. "
                       "Default is `%s`." % DT_FORMAT_FILE)

argp.add_argument('-v', '--validation-set', type=str, dest="validate", nargs="*",
                  help="The list of data files to be used for validation. ")
argp.add_argument('--validation-format', type=str, dest="validate_format", required=False,
                  help="The format to be used when loading validation set. Default is same as data format.")

argp.add_argument('-d', '--delimiter', type=str, required=False, metavar="delimiter", default=None,
                  dest="delimiter", help="The delimiter to be expected in the data files. Default is whitespace.")
argp.add_argument('-b', '--batch', type=int, required=False, metavar="batch_size", default=DEF_BATCH, dest="batch_size",
                  help="The size of the mini-batches. Default is %d." % DEF_BATCH)

argp.add_argument('-m', '--model-path', type=str, metavar="model_path", dest="model_path", required=True,
                  help="The path to the model - both for retrieving and storing.")
argp.add_argument('-c', '--checkpoint', type=int, dest="checkpoint_secs", default=DEF_CHECKPOINT_SECS,
                  help="Number of seconds between checkpoint saves. "
                       "Default is %d" % DEF_CHECKPOINT_SECS)
argp.add_argument('-s', '--summary', type=int, dest="summary_steps", default=DEF_SUMMARY_STEPS,
                  help="The number of steps between summary dump and validation test. "
                       "Default is %d" % DEF_SUMMARY_STEPS)

argp.add_argument('-y', '--early', type=int, required=False, metavar="early_stopping_rounds", dest="early",
                  default=None, help="How many steps the loss should not change to decide early stopping. "
                                     "Default is None, i.e. - no early stopping")
argp.add_argument('-e', '--epochs', type=int, required=False, metavar="epochs", default=None, dest="epochs",
                  help="The number of epochs the test to be run. Default is Infinity.")

argp.add_argument('-q', '--quiet', required=False, action="store_true", dest="quite",
                  help="Whether to suppress the more detailed info messages.")

args = argp.parse_args()

# Sanity checks
if not args.quite:
    tf.logging.set_verbosity(tf.logging.INFO)

# Restore the model
input_x = tf_prepare_pipe(args.train,
                          data_format=args.format,
                          delimiter=args.delimiter,
                          num_epochs=args.epochs,
                          batch_size=args.batch_size,
                          shuffle=True,
                          allow_smaller_batches=False,
                          name="input_pipe")
model, saver, model_path, _ = \
    tf_restore_graph(args.model_path,
                     input_pipe=input_x)

assert model is not None
if input_x is not None:
    assert model.batch_size == args.batch_size
else:
    args.batch_size = model.batch_size

# Now prepare the hooks, needed for the training runs.
hooks = [tf.train.NanTensorHook(model.loss_op)]

# Deal with the input
if args.validate is not None:
    if args.validate_format is None:
        args.validate_format = args.format

    # Some format changes for clarity.
    if args.validate_format == TF_FORMAT_PIPE:
        args.validate_format = DT_FORMAT_FILE
    elif args.validate_format == TF_FORMAT_CONST:
        args.validate_format = DT_FORMAT_MEM

    tf_logging.info("Preparing the validation data...")
    val_data = dt_prepare_iterator(args.validate, data_format=args.validate_format,
                                   delimiter=args.delimiter,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_epochs=1)
    val_hook = ValidationHook(model,
                              data_iterator=val_data,
                              every_n_steps=args.summary_steps, early_stopping_rounds=args.early,
                              report_fn=lambda cnt, loss: tf_logging.info("Validation loss: %.9f @ %d" % (loss, cnt)))
    hooks.append(val_hook)
    tf_logging.info("... done.")
else:
    if args.early is not None:
        argp.error("You can't specify early stopping steps, if there is no validation set specified!")
    val_hook = None

# Now arrange the other methods of input
if input_x is None and args.train is not None:
    input_x = dt_prepare_iterator(args.train, data_format=args.format,
                                  delimiter=args.delimiter,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_epochs=args.epochs,
                                  allow_smaller_batch=True)

    tf_logging.info("Preparing the training data feed...")
    hooks.append(FeedDataHook(model.input_var, iterator=input_x))
    tf_logging.info("... done.")

if not args.quite:
    hooks.append(TrainLogHook(model))

# Now come the actual training loop
loss = None
with tf.train.MonitoredTrainingSession(is_chief=True,
                                       scaffold=tf.train.Scaffold(saver=saver),
                                       chief_only_hooks=hooks,
                                       checkpoint_dir=os.path.dirname(model_path),
                                       save_checkpoint_secs=args.checkpoint_secs,
                                       save_summaries_steps=args.summary_steps) as sess:
    while not sess.should_stop():
        _, loss = sess.run((model.train_op, model.loss_op))

tf_logging.info("Training finished w/ last training loss = %.9f" % loss)
if val_hook is not None:
    tf_logging.info("  ... best step: %d, w/ validation loss = %.9f)" %
                    (val_hook.best_step, val_hook.best_loss)
                    )
