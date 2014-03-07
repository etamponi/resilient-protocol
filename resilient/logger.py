import os
import sys
import cPickle

import numpy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class Logger(object):
    def __init__(self):
        numpy.set_printoptions(formatter={
            'float_kind': self.format_number,
            'int_kind': self.format_number
        })
        self.terminal = sys.stdout
        self.log_string = ""
        self.disable_log = False

    def write(self, message):
        self.terminal.write(message)
        if message.startswith("\r"):
            self.disable_log = True
        if not self.disable_log:
            self.log_string += message
        if message.endswith("\n") and self.disable_log:
            self.disable_log = False
        self.flush()

    def finish(self, log_file, **data):
        sys.stdout = self.terminal
        self._ensure_dir(log_file)
        with open(log_file + ".txt", "w") as f:
            f.write(self.log_string)
        if len(data) > 0:
            with open(log_file + ".dat", "w") as f:
                cPickle.dump(data, f)

    def flush(self):
        self.terminal.flush()

    @staticmethod
    def _ensure_dir(log_file):
        d = os.path.dirname(log_file)
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def format_number(x):
        return "{:5.3f}".format(x)
