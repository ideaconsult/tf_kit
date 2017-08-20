#
# TensorFlow training and running routines and classes.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev
#
"""
The TensorFlow gears defines these classes:

@@FeedDataHook
@@TrainLogHook
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
    def __init__(self, feed_op, iterator, bridge_op=None, input_op=None, batch_size=None):
        """
        An implementation for reading data and feeding it to the given `feed_op`.
        :param feed_op: The feed op to referenced.
        :param iterator: The iterator to be used for retrieving the data.
        :param bridge_op: If passed the input is taken as the result of this op and
                        the input_op is used to feed the data for it.
        :param input_op: If provided the data is actually taken from here, or if
                        `bridge_op` is passed - this is actually the input for
                        the bridging architecture which produces the output.
        """
        assert iterator is not None or input_op is not None
        self._iterator = iterator
        self._input_op = input_op
        self._bridge_op = bridge_op
        self._batch_size = iterator.batch_size if iterator is not None else batch_size
        self._feed_op = feed_op

    def before_run(self, run_context):
        x = None
        if self._iterator is not None:
            x = self._iterator.next()
        elif self._bridge_op is not None:
            x = run_context.session.run(self._bridge_op, feed_dict={self._input_op: x})
        else:
            x = run_context.session.run(self._input_op)

        if x.shape[0] < self._batch_size:
            padding = self._batch_size - x.shape[0]
            x = np.append(x, np.array([[.0] * x.shape[1]] * padding), axis=0)
        return tf.train.SessionRunArgs(None, {self._feed_op: x})

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


class ValidationHook:
    def __init__(self, model,
                 data_iterator,
                 every_n_steps=50,
                 early_stopping_rounds=10,
                 report_fn=None):
        """

        :param model: The dictionary / object to retrieve `loss_op` and `input_var` tensors. 
        :param data_iterator: The input data for the validation runs.
        :param every_n_steps: On how many steps to perform validation.
        :param early_stopping_rounds: After how many validation rounds without loss improvement to call
                for early stopping.
        :param report_fn: The function to report each validation result to: { lambda counter, loss } 
        """
        self._model = model
        self._iterator = data_iterator
        self._every_n_steps = every_n_steps
        self._early_stopping_rounds = early_stopping_rounds
        self._report_fn = report_fn
        self._last_round = 0
        self._counter = 0
        self._best_rounds = 0
        self._best_loss = None
        self._best_step = 0

    def _run_validation(self, sess):
        return tf_validation_run(sess, self._model, self._iterator)

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        self._counter = run_context.session.run(tf.train.get_global_step())
        if self._counter - self._last_round < self._every_n_steps:
            return None

        self._last_round = self._counter
        loss = self._run_validation(run_context.session)
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss
            self._best_rounds = 0
            self._best_step = self._counter
        else:
            self._best_rounds += 1

        if self._report_fn is not None:
            self._report_fn(self._counter, loss)

        if self._early_stopping_rounds is not None and self._early_stopping_rounds <= self._best_rounds:
            tf_logging.info("Early stopping at step %04d, with loss: %.9f" % (self._counter, loss))
            run_context.request_stop()

    def begin(self):
        pass

    def end(self, session):
        pass

    def after_create_session(self, sess, coord):
        loss = self._run_validation(sess)
        tf_logging.info("Initial validation loss: %.9f" % loss)

    @property
    def best_step(self):
        return self._best_step

    @property
    def best_loss(self):
        return self._best_loss  if self._best_loss is not None else np.nan
