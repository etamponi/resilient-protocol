"""
Wraps the Multilayer Perceptron implementation present in PyBrain so that it can
be used as a scikit-learn estimator.
"""
from abc import ABCMeta, abstractmethod
from itertools import izip
import numpy

from ffnet.ffnet import ffnet, mlgraph, imlgraph

from pybrain.datasets.importance import ImportanceDataSet
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing.data import OneHotEncoder
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
            p = self.activate(x)
            if p.min() < 0:
                p = p - p.min()
            p = p / p.sum()
            probs[i] = p
        preprocessing.normalize(probs, norm="l1", copy=False)
        return probs

    def predict(self, inp):
        p = self.predict_proba(inp)
        return self.classes_[numpy.argmax(p, axis=1)]


class PyBrainNetwork(ANNWrapper):

    def __init__(self, hidden_neurons=None, output_class="softmax",
                 learning_rate=0.01, lr_decay=1.0, momentum=0.0,
                 weight_decay=0.0, max_epochs=None, continue_epochs=10,
                 validation_percent=0.25, fast=True, random_state=None):
        super(PyBrainNetwork, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.output_class = output_class
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
            hidden_neurons = (n_features + self.n_classes_)/2
        else:
            hidden_neurons = self.hidden_neurons
        self.network_ = buildNetwork(
            n_features, hidden_neurons, self.n_classes_,
            outclass=self._get_output_class(), fast=self.fast
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

    def _get_output_class(self):
        if self.output_class == "softmax":
            return SoftmaxLayer
        elif self.output_class == "tanh":
            return TanhLayer
        elif self.output_class == "sigmoid":
            return SigmoidLayer
        elif self.output_class == "linear":
            return LinearLayer
        else:
            raise ValueError(
                "output_class can be: softmax, tanh, sigmoid, linear"
            )


class FFNetNetwork(ANNWrapper):

    def __init__(self, hidden_neurons="a", bias=True,
                 independent_outputs=False, training_fn="bfgs",
                 training_params=None, random_state=None):
        super(FFNetNetwork, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.bias = bias
        self.independent_outputs = independent_outputs
        self.training_fn = training_fn
        self.training_params = training_params
        self.random_state = random_state
        # Training time attributes
        self.network_ = None

    def fit(self, inp, y, sample_weight=None):
        self.classes_, y = numpy.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        if self.training_params is None:
            training_params = {}
        else:
            training_params = self.training_params

        y = OneHotEncoder().fit_transform(array2d(y).transpose()).toarray()
        n_features = inp.shape[1]
        random_state = check_random_state(self.random_state)

        if self.hidden_neurons == "a":
            hidden_neurons = (n_features + self.n_classes_)/2
        else:
            hidden_neurons = self.hidden_neurons
        if self.independent_outputs:
            graph = imlgraph(
                (n_features, hidden_neurons, self.n_classes_), biases=self.bias
            )
        else:
            graph = mlgraph(
                (n_features, hidden_neurons, self.n_classes_), biases=self.bias
            )
        self.network_ = ffnet(graph, random_state)
        self.network_.randomweights()
        trainer = getattr(self.network_, "train_" + self.training_fn)
        trainer(inp, y, **training_params)
        return self

    def activate(self, x):
        return self.network_(x)
