"""
Wraps the Multilayer Perceptron implementation present in PyBrain so that it can
be used as a scikit-learn estimator.
"""
from abc import ABCMeta, abstractmethod
from itertools import izip
import numpy
from pybrain.datasets.importance import ImportanceDataSet
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import array2d

__author__ = 'Emanuele Tamponi <emanuele.tamponi@gmail.com>'


class ANNWrapper(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.classes_ = None
        self.n_classes_ = None

    @abstractmethod
    def fit(self, inp, y, sample_weight=None):
        pass

    @abstractmethod
    def activate(self, x):
        pass

    def predict_proba(self, inp):
        inp = array2d(inp)
        probs = numpy.zeros((len(inp), self.n_classes_))
        for i, x in enumerate(inp):
            probs[i] = self.activate(x)
        preprocessing.normalize(probs, norm="l1", copy=False)
        return probs

    def predict(self, inp):
        p = self.predict_proba(inp)
        return self.classes_[numpy.argmax(p, axis=1)]


class PyBrainNetwork(ANNWrapper):

    def __init__(self, hidden_neurons=None, learning_rate=0.01, lr_decay=1.0,
                 momentum=0.0, weight_decay=0.0, max_epochs=None,
                 continue_epochs=10, validation_percent=0.25, fast=True,
                 random_state=None):
        super(PyBrainNetwork, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.continue_epochs = continue_epochs
        self.validation_percent = validation_percent
        self.random_state = random_state
        self.fast = fast
        self.network_ = None
        self.trainer_ = None

    def fit(self, inp, y, sample_weight=None):
        self.classes_, y = numpy.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        n_features = inp.shape[1]
        random_state = check_random_state(self.random_state)

        # We need to build an ImportanceDataSet from inp, y and sample_weight
        dataset = ImportanceDataSet(n_features, self.n_classes_)
        if sample_weight is None:
            sample_weight = numpy.ones(len(y))
        for x, label_pos, weight in izip(inp, y, sample_weight):
            target = numpy.zeros(self.n_classes_)
            target[label_pos] = 1
            weight = weight * numpy.ones(self.n_classes_)
            dataset.newSequence()
            dataset.addSample(x, target, weight)

        if self.hidden_neurons is None:
            self.network_ = buildNetwork(
                n_features, (n_features + self.n_classes_)/2, self.n_classes_,
                outclass=SoftmaxLayer, fast=self.fast
            )
        else:
            self.network_ = buildNetwork(
                n_features, self.hidden_neurons, self.n_classes_,
                outclass=SoftmaxLayer, fast=self.fast
            )

        # Set the initial parameters in a repeatable way
        net_params = random_state.random_sample(self.network_.paramdim)
        self.network_.params[:] = net_params

        self.trainer_ = BackpropTrainer(
            self.network_, dataset=dataset, learningrate=self.learning_rate,
            lrdecay=self.lr_decay, momentum=self.momentum,
            weightdecay=self.weight_decay
        )
        self.trainer_.trainUntilConvergence(
            random_state, maxEpochs=self.max_epochs,
            continueEpochs=self.continue_epochs,
            validationProportion=self.validation_percent
        )
        return self

    def activate(self, x):
        return self.network_.activate(x)


class FFNetNetwork(ANNWrapper):

    def __init__(self):
        super(FFNetNetwork, self).__init__()

    def fit(self, inp, y, sample_weight=None):
        pass

    def activate(self, x):
        pass