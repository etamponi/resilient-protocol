__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class EstimatorWrapper(object):

    def __init__(self, estimator, train_set, test_set):
        self._content = estimator
        self._train_set = train_set
        self._test_set = test_set

    def __getattr__(self, item):
        return self._content.__getattribute__(item)

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set
