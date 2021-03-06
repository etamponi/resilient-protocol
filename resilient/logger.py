from multiprocessing import Lock
import os
import cPickle

import numpy


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class Logger(object):

    logger = None

    @classmethod
    def get(cls):
        if cls.logger is None:
            cls.logger = Logger()
        return cls.logger

    def __init__(self):
        if Logger.logger is not None:
            print "Error: a Logger already exists"
            exit(1)
        numpy.set_printoptions(formatter={
            'float_kind': self.format_number,
            'int_kind': self.format_number
        })
        self.log_string = ""
        self.lock = Lock()

    def write(self, *tokens):
        self.lock.acquire()
        message = " ".join(str(token) for token in tokens)
        if message.startswith("!"):
            message = "{:6d}: {}".format(os.getpid(), message[1:])
        else:
            self.log_string += message + "\n"
        print message
        self.lock.release()

    @staticmethod
    def dump(filename, **data):
        Logger._ensure_dir(filename)
        if len(data) > 0:
            with open(filename + ".dat", "w") as f:
                cPickle.dump(data, f)

    def save(self, log_name, **data):
        self._ensure_dir(log_name)
        with open(log_name + ".txt", "w") as f:
            f.write(self.log_string)
        self.dump(log_name, **data)

    def clear(self):
        self.log_string = ""

    @staticmethod
    def _ensure_dir(log_file):
        d = os.path.dirname(log_file)
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def format_number(x):
        return "{:5.3f}".format(x)
