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
    epilog="The model is extracted from the given location, searching for tensors and ops from the "
           "designated collections: inputs, outputs, train_ops, losses, targets, latents, generators. "
           "The last three are optional."
)

argp.add_argument('-t', '--training-set', type=str, dest="train", nargs='*',
                  help="The list of data files to be used for training.")
argp.add_argument('-v', '--validation-set', type=str, dest="validate", nargs="*",
                  help="The list of data files to be used for validation. ")
argp.add_argument('-d', '--delimiter', type=str, required=False, metavar="delimiter", default=None,
                  dest="delimiter", help="The delimiter to be expected in the data files. Default is whitespace.")
argp.add_argument('-b', '--batch', type=int, required=False, metavar="batch_size", default=None, dest="batch_size",
                  help="The size of the mini-batches. Default is derived from the model.")

argp.add_argument('-m', '--model-dir', type=str, metavar="model_dir", dest="model_dir", required=True,
                  help="The path to the model - both for retrieving and storing.")
argp.add_argument('-c', '--checkpoint', type=int, dest="checkpoint_secs", default=DEF_CHECKPOINT_SECS,
                  help="Number of seconds between checkpoint saves. "
                       "Default is %d" % DEF_CHECKPOINT_SECS)
argp.add_argument('-i', '--model-index', type=int, dest="model_index", default=0,
                  help="The index in the model collection to retrieve it from. Default is 0.")

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

# First, restore the model and align the batch_size
model, saver, _, _ = tf_restore_graph(args.model_dir)
assert model is not None

if args.batch_size is None:
    args.batch_size = model.batch_size
assert args.batch_size > 0

# Now make the train and validate trainset more flexible in setting.
if len(args.train) == 1:
    args.train = args.train[0]
    delimiter = args.delimiter or dt_delimiter_from_ext(args.train)
else:
    delimiter = args.delimiter or " "

if args.validate is not None and len(args.validate) == 1:
    args.validate = args.validate[0]

# Now prepare the hooks, needed for the training runs.
hooks = [tf.train.NanTensorHook(model.loss_op)]

# Deal with the input
if args.validate is not None:
    tf_logging.info("Preparing the validation data...")
    val_data = dt_prepare_iterator(args.validate,
                                   delimiter=delimiter,
                                   skip_rows=0,
                                   batch_size=args.batch_size,
                                   allow_smaller_batch=False,
                                   num_epochs=1,
                                   shuffle=False)
    val_hook = ValidationHook(model,
                              data_iterator=val_data,
                              every_n_steps=args.summary_steps,
                              early_stopping_rounds=args.early,
                              report_fn=lambda cnt, loss: tf_logging.info("Validation loss: %.9f @ %d" % (loss, cnt)))
    hooks.append(val_hook)
    tf_logging.info("... done.")
else:
    if args.early is not None:
        argp.error("You can't specify early stopping steps, if there is no validation set specified!")
    val_hook = None

# Now arrange the other methods of input, if such are passed.
if args.train is not None:
    input_x = dt_prepare_iterator(args.train,
                                  delimiter=delimiter,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  skip_rows=0,
                                  num_epochs=args.epochs,
                                  allow_smaller_batch=False)

    tf_logging.info("Preparing the training data feed...")
    hooks.append(FeedDataHook(feed_op=model.input_var, iterator=input_x))
    tf_logging.info("... done.")

if not args.quite:
    hooks.append(TrainLogHook(model))

# Now come the actual training loop
loss = None
with tf.train.MonitoredTrainingSession(is_chief=True,
                                       scaffold=tf.train.Scaffold(saver=saver),
                                       chief_only_hooks=hooks,
                                       checkpoint_dir=args.model_dir,
                                       save_checkpoint_secs=args.checkpoint_secs,
                                       save_summaries_steps=args.summary_steps) as sess:
    while not sess.should_stop():
        _, loss = sess.run((model.train_op, model.loss_op))

tf_logging.info("Training finished w/ last training loss = %.9f" % loss)
if val_hook is not None:
    tf_logging.info("  ... best step: %d, w/ validation loss = %.9f)" %
                    (val_hook.best_step, val_hook.best_loss)
                    )
