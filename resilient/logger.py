import sys

import numpy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class Logger(object):
    def __init__(self, filename):
        numpy.set_printoptions(formatter={'float': lambda x: "{:5.3f}".format(x), 'int': lambda x: "{:5d}".format(x)})
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.disable_log = False

    def write(self, message):
        self.terminal.write(message)
        if message.startswith("\r"):
            self.disable_log = True
        if not self.disable_log:
            self.log.write(message)
        if message.endswith("\n") and self.disable_log:
            self.disable_log = False
        self.flush()

    def finish(self):
        sys.stdout = self.terminal

    def flush(self):
        self.log.flush()
        self.terminal.flush()
