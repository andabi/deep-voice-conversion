# -*- coding: utf-8 -*-
#!/usr/bin/env python

import re
from tensorpack.utils import logger
from tensorpack.tfutils.gradproc import GradientProcessor
from tensorpack.callbacks.monitor import JSONWriter
import tensorflow as tf


# class AudioWriter(TrainingMonitor):
#     """
#     Write summaries to TensorFlow event file.
#     """
#     def __new__(cls):
#         if logger.get_logger_dir():
#             return super(TFEventWriter, cls).__new__(cls)
#         else:
#             logger.warn("logger directory was not set. Ignore TFEventWriter.")
#             return NoOpMonitor()
#
#     def _setup_graph(self):
#         self._writer = tf.summary.FileWriter(logger.get_logger_dir(), graph=tf.get_default_graph())
#
#     def process_summary(self, summary):
#         self._writer.add_summary(summary, self.global_step)
#
#     def process_event(self, evt):
#         self._writer.add_event(evt)
#
#     def _trigger(self):     # flush every epoch
#         self._writer.flush()
#
#     def _after_train(self):
#         self._writer.close()
#

class FilterGradientVariables(GradientProcessor):
    """
    Skip the update of certain variables and print a warning.
    """

    def __init__(self, var_regex='.*', verbose=True):
        """
        Args:
            var_regex (string): regular expression to match variable to update.
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterGradientVariables, self).__init__()
        self._regex = var_regex
        self._verbose = verbose

    def _process(self, grads):
        g = []
        to_print = []
        for grad, var in grads:
            if re.match(self._regex, var.op.name):
                g.append((grad, var))
            else:
                to_print.append(var.op.name)
        if self._verbose and len(to_print):
            message = ', '.join(to_print)
            logger.warn("No gradient w.r.t these trainable variables: {}".format(message))
        return g
