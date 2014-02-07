__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class Dataset(object):

    def __init__(self, inp, y):
        self._data = inp
        self._target = y

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    def __iter__(self):
        for instance, target in zip(self._data, self._target):
            yield instance, target
