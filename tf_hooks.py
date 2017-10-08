#
# TensorFlow training and running routines and classes.
#
# Copyright (C) 2017, IDEAConsult Ltd.
# Author: Ivan (Jonan) Georgiev
#
"""
The TensorFlow hooks which are defined here:

@@FeedDataHook
@@TrainLogHook
@@ProbabilisticOpsHook
@@ValidationHook

"""

from .tf_utils import *
from .dt_constants import DEF_DISPLAY_STEP
from tensorflow.python.platform import tf_logging
import numpy as np


class FeedDataHook:
    """
    A MonitoredSession hook capable of providing data to the model's input variable.
    """
    def __init__(self, iterator=None, input_op=None, batch_size=None, feed_op=None, feed_fn=None):
        """
        An implementation for reading data and feeding it to the given `feed_op`.
        :param iterator: The iterator to be used for retrieving the data.
        :param input_op: If provided the data is actually taken from here.
        :param batch_size: The size of the batches, if iterator is not provided to help.
        :param feed_op: The feed op to referenced.
        :param feed_fn: The pre-processing function of format (sess, input) -> {feed dict}
        """
        assert iterator is not None or input_op is not None
        assert feed_op is not None or feed_fn is not None

        self._iterator = iterator
        self._input_op = input_op
        self._feed_fn = feed_fn
        self._feed_op = feed_op
        self._batch_size = iterator.batch_size if iterator is not None else batch_size

    def before_run(self, run_context):
        x = run_context.session.run(self._input_op) if self._iterator is None else self._iterator.next()

        if x.shape[0] < self._batch_size:
            padding = self._batch_size - x.shape[0]
            x = np.append(x, np.array([[.0] * x.shape[1]] * padding), axis=0)

        return tf.train.SessionRunArgs(None,
                                       {self._feed_op: x} if self._feed_fn is None
                                       else self._feed_fn(run_context.session, x))

    def after_run(self, run_context, run_values):
        pass

    def begin(self):
        pass

    def end(self, session):
        pass

    def after_create_session(self, sess, coord):
        pass


class TrainLogHook:
    def __init__(self, model,
                 log_step=DEF_DISPLAY_STEP,
                 logarithmic_log=True,
                 before_fn=None,
                 after_fn=None,
                 done_fn=None):
        """

        :param model: The model to get the loss op from. 
        :param log_step: On how many steps to log the training loss.
        :param logarithmic_log: If True - every time the log_step doubles.
        :param before_fn: The function to be executed before every session run, 
                {lambda session, counter: }
        :param after_fn: The function to be executed after every session run. 
                {lambda session, counter, loss: }
        :param done_fn: The function to be called at the end, before the session is closed: 
                { lambda: session, counter } 
        """
        self._model = model
        self._counter = 0
        self.log_step = log_step
        self._before_fn = before_fn
        self._after_fn = after_fn
        self._done_fn = done_fn
        self._logarithmic_log = logarithmic_log

    def before_run(self, run_context):
        self._counter = run_context.session.run(tf.train.get_global_step())
        if self._before_fn is not None:
            self._before_fn(run_context.session, self._counter)
        return tf.train.SessionRunArgs(self._model.loss_op)

    def after_run(self, run_context, run_values):
        loss = run_values.results

        self._counter += 1
        if self._after_fn is not None:
            self._after_fn(run_context.session, self._counter, loss)

        if self._counter % self.log_step == 1:
            if self._logarithmic_log:
                self.log_step *= 2
            tf_logging.info("Step: %04d, training loss: %.9f" % (self._counter, loss))

    def begin(self):
        trainables = tf.trainable_variables()
        tf_logging.info(
            "Training of %d tensors with total of %d parameters started..." % (
                len(trainables),
                tf_tensors_size(trainables, collapse=True)
            ))

    def end(self, session):
        if self._done_fn is not None:
            self._done_fn(session, self._counter + 1)

    def after_create_session(self, sess, coord):
        pass


class ProbabilisticOpsHook:
    def __init__(self, ops, initial_probs, ensure_non_empty=False, exclusive=False):
        """

        :param models: A list of models to balance training to.
        :param initial_probs: The initial probabilities for each op.
        :param ensure_non_empty: Make sure at least one is executed.
        :param exclusive: Allow only one train op to be executed
        """
        self._ops = ops
        self.probabilities = initial_probs
        self._non_empty = ensure_non_empty
        self._exclusive = exclusive

    def before_run(self, run_context):
        fetches = []
        prob = np.random.random()
        for i in range(len(self._ops)):
            if self.probabilities[i] > prob:
                fetches.append(self._ops[i])
                if self._exclusive:
                    break

        # Ensure at least one - the most probable.
        if not fetches and self._non_empty:
            fetches.append(self._ops[np.argmax(self.probabilities)])
        return tf.train.SessionRunArgs(fetches)

    def after_run(self, run_context, run_values):
        pass

    def begin(self):
        pass

    def end(self, session):
        pass

    def after_create_session(self, sess, coord):
        pass


class ValidationHook:
    def __init__(self, model,
                 data_iterator,
                 every_n_steps=50,
                 early_stopping_rounds=10,
                 report_fn=None,
                 feed_fn=None):
        """
        A hook for performing validation step.
        :param model: The dictionary / object to retrieve `loss_op` and `input_var` tensors. 
        :param data_iterator: The input data for the validation runs.
        :param every_n_steps: On how many steps to perform validation.
        :param early_stopping_rounds: After how many validation rounds without loss improvement to call
                for early stopping.
        :param report_fn: The function to report each validation result to: { lambda counter, loss }
        :param feed_fn: The pre-processing function of format (sess, input) -> {feed dict}

        """
        self._model = model
        self._iterator = data_iterator
        self._every_n_steps = every_n_steps
        self._early_stopping_rounds = early_stopping_rounds
        self._report_fn = report_fn
        self._feed_fn = feed_fn
        self._last_round = 0
        self._last_loss = np.nan
        self._counter = 0
        self._best_rounds = 0
        self._best_loss = None
        self._best_step = 0

    def _run_validation(self, sess):
        return tf_validation_run(sess, self._model, self._iterator, self._feed_fn)

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        self._counter = run_context.session.run(tf.train.get_global_step())
        if self._counter - self._last_round < self._every_n_steps:
            return None

        self._last_round = self._counter
        self._last_loss = self._run_validation(run_context.session)
        if self._best_loss is None or self._last_loss < self._best_loss:
            self._best_loss = self._last_loss
            self._best_rounds = 0
            self._best_step = self._counter
        else:
            self._best_rounds += 1

        if self._report_fn is not None:
            self._report_fn(self._counter, self._last_loss)

        if self._early_stopping_rounds is not None and self._early_stopping_rounds <= self._best_rounds:
            if self._report_fn is not None:
                self._report_fn(self._counter, self._last_loss, True)
            run_context.request_stop()

    def begin(self):
        pass

    def end(self, session):
        pass

    def after_create_session(self, sess, coord):
        loss = self._run_validation(sess)
        if self._report_fn is not None:
            self._report_fn(0, loss)

    @property
    def best_step(self):
        return self._best_step

    @property
    def best_loss(self):
        return self._best_loss if self._best_loss is not None else np.nan

    @property
    def last_loss(self):
        return self._last_loss
