import os
import sys

import numpy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class Logger(object):
    def __init__(self, filename):
        numpy.set_printoptions(formatter={'float': lambda x: "{:5.3f}".format(x), 'int': lambda x: "{:5d}".format(x)})
        self.terminal = sys.stdout
        self.log_file = filename
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

    def finish(self):
        sys.stdout = self.terminal
        self._ensure_dir()
        with open(self.log_file, "w") as f:
            f.write(self.log_string)

    def flush(self):
        self.terminal.flush()

    def _ensure_dir(self):
        d = os.path.dirname(self.log_file)
        if not os.path.exists(d):
            os.makedirs(d)
