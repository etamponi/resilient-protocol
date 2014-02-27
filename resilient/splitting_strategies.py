from cmath import log
from copy import deepcopy
from itertools import izip, product
from abc import ABCMeta, abstractmethod

import numpy
from scipy.spatial import distance
from sklearn.base import BaseEstimator
from sklearn.preprocessing.data import MinMaxScaler

from resilient.dataset import Dataset
from resilient.pdfs import DistanceExponential


__author__ = 'Emanuele Tamponi <emanuele.tamponi@diee.unica.it>'


class SplittingStrategy(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def iterate(self, inp, y, random_state):
        pass

    @staticmethod
    def _make_datasets(inp, y, train_indices, test_indices):
        train_data, test_data = inp[train_indices], inp[test_indices]
        train_targ, test_targ = y[train_indices], y[test_indices]
        return Dataset(train_data, train_targ), Dataset(test_data, test_targ)


class CentroidBasedPDFSplittingStrategy(SplittingStrategy):

    def __init__(self, n_estimators=101, pdf=DistanceExponential(), train_percent=0.35, replace=False, repeat=False):
        self.n_estimators = n_estimators
        self.pdf = pdf
        self.train_percent = train_percent
        self.replace = replace
        self.repeat = repeat

    def iterate(self, inp, y, random_state):
        print "\rTraining", self.n_estimators, "estimators..."
        for probs in self._get_probabilities(inp, random_state):
            train_indices, test_indices = self._make_indices(len(inp), probs, random_state)
            yield self._make_datasets(inp, y, train_indices, test_indices)

    def _make_indices(self, l, pdf, random_state):
        train_indices = random_state.choice(l, size=int(self.train_percent*l), p=pdf, replace=self.replace)
        if not self.repeat:
            train_indices = numpy.unique(train_indices)
        test_indices = numpy.ones(l, dtype=bool)
        test_indices[train_indices] = False
        return train_indices, test_indices

    def _get_probabilities(self, inp, random_state):
        #inp = MinMaxScaler(feature_range=(0, 1)).fit_transform(inp)
        mean_probs = numpy.ones(inp.shape[0]) / inp.shape[0]
        for i in range(self.n_estimators):
            mean = inp[random_state.choice(len(inp), p=mean_probs)]
            probs = self.pdf.probabilities(inp, mean=mean)
            for j, x in enumerate(inp):
                mean_probs[j] *= log(1 + distance.euclidean(x, mean)).real
            mean_probs = mean_probs / mean_probs.sum()
            yield probs


class CentroidBasedKNNSplittingStrategy(SplittingStrategy):

    def __init__(self, n_estimators=101, train_percent=0.25):
        self.n_estimators = n_estimators
        self.train_percent = train_percent

    def iterate(self, inp, y, random_state):
        print "\rTraining", self.n_estimators, "estimators..."
        k = int(self.train_percent * len(inp))
        for distances in self._get_distances(inp, random_state):
            indices = distances.argsort()
            train_indices, test_indices = indices[:k], indices[k:]
            yield self._make_datasets(inp, y, train_indices, test_indices)

    def _get_distances(self, inp, random_state):
        mean_probs = numpy.ones(inp.shape[0]) / inp.shape[0]
        distances = numpy.zeros(inp.shape[0])
        for i in range(self.n_estimators):
            mean = inp[random_state.choice(len(inp), p=mean_probs)]
            for j, x in enumerate(inp):
                distances[j] = distance.euclidean(x, mean)
                mean_probs[j] *= distances[j]
            mean_probs = mean_probs / mean_probs.sum()
            yield distances


class GridSplittingStrategy(SplittingStrategy):

    def __init__(self, n_cells_per_dim=4, min_cell_size=10, k_overlap=10, dist_measure=distance.cityblock):
        self.n_cells_per_dim = n_cells_per_dim
        self.min_cell_size = min_cell_size
        self.k_overlap = k_overlap
        self.dist_measure = dist_measure

    def iterate(self, inp, y, random_state):
        cells = self._get_cells(inp, y)
        avg = float(sum([len(v[0]) for v in cells.values()])) / len(cells)
        print "\rTraining", len(cells), "estimators on an average of", avg, "examples"
        for source, target in cells.values():
            source, target = numpy.array(source), numpy.array(target)
            yield Dataset(source, target), Dataset(source, target)

    def _get_cells(self, inp, y):
        s = MinMaxScaler(feature_range=(0, self.n_cells_per_dim)).fit(inp)
        cells = {}
        for x, targ in izip(inp, y):
            cell_code = tuple([int(t-0.001) for t in s.transform(x)])
            if cell_code not in cells:
                cells[cell_code] = [[], []]
            cells[cell_code][0].append(x)
            cells[cell_code][1].append(targ)
        smallest_cell, smallest_size = self._get_smallest_cell(cells)
        while smallest_size < self.min_cell_size:
            print "\rSmallest cell:", smallest_cell, "- size:", smallest_size,
            other_cell = self._get_nearest_cell_code(smallest_cell, cells)
            cells[other_cell][0].extend(cells[smallest_cell][0])
            cells[other_cell][1].extend(cells[smallest_cell][1])
            del cells[smallest_cell]
            smallest_cell, smallest_size = self._get_smallest_cell(cells)
        print ""
        return self._group_cells(cells)

    def _group_cells(self, cells):
        final_cells = deepcopy(cells)
        for i, cell_code in enumerate(cells):
            print "\rExpanding cell:", (i+1), "of", len(cells),
            neighbor_cells = self._get_neighbor_cells(cell_code, cells)
            for neighbor_code in neighbor_cells:
                final_cells[cell_code][0].extend(cells[neighbor_code][0])
                final_cells[cell_code][1].extend(cells[neighbor_code][1])
        print ""
        return final_cells

    @staticmethod
    def _get_smallest_cell(cells):
        smallest_cell_size = numpy.iinfo(numpy.int32).max
        smallest_cell_code = None
        for cell_code, (source, target) in cells.iteritems():
            if len(source) < smallest_cell_size:
                smallest_cell_code = cell_code
                smallest_cell_size = len(source)
        return smallest_cell_code, smallest_cell_size

    def _get_nearest_cell_code(self, cell_code, cells):
        nearest_cell_distance = numpy.iinfo(numpy.int32).max
        nearest_cell_code = None
        for other_cell_code in cells:
            if other_cell_code == cell_code:
                continue
            d = self.dist_measure(cell_code, other_cell_code)
            if d < nearest_cell_distance:
                nearest_cell_code = other_cell_code
                nearest_cell_distance = d
        return nearest_cell_code

    def _get_neighbor_cells(self, cell_code, cells):
        cells = numpy.array(cells.keys())
        distances = numpy.array([self.dist_measure(cell_code, other_code) for other_code in cells])
        indices = distances.argsort()
        cells = list(cells[indices[1:self.k_overlap+1]])
        cells = [tuple(cell) for cell in cells]
        return cells


class SquareGridSplittingStrategy(SplittingStrategy):

    def __init__(self, spacing=0.5, overlapping_radius=5, cell_dist_measure=distance.cityblock):
        self.spacing = spacing
        self.overlapping_radius = overlapping_radius
        self.cell_dist_measure = cell_dist_measure

    def iterate(self, inp, y, random_state):
        cells = self._get_cells(inp, y)
        neighbors = self._get_neighbors(cells)
        non_redundant_cells = self._get_non_redundant_cells(cells, neighbors)
        cells = self._overlap_cells(non_redundant_cells, cells, neighbors)
        avg = float(sum([len(v[0]) for v in cells.values()])) / len(cells)
        print "\rTraining", len(cells), "estimators on an average of", avg, "examples"
        for source, target in cells.values():
            source, target = numpy.array(source), numpy.array(target)
            yield Dataset(source, target), Dataset(source, target)

    @staticmethod
    def _get_cells(inp, ys):
        # minimum = numpy.min(inp, axis=0)
        # inp_flt = numpy.zeros_like(inp)
        # for i in xrange(len(inp_flt)):
        #     inp_flt[i] = (inp[i] + minimum) / self.spacing
        cells = {}
        # for x_flt, x, y in izip(inp_flt, inp, ys):
        #     code = tuple([floor(t) for t in x_flt])
        for x, y in izip(inp, ys):
            code = tuple([int(t) for t in x])
            if code not in cells:
                cells[code] = ([], [])
            cells[code][0].append(x)
            cells[code][1].append(y)
        print "\rCells found:", len(cells)
        return cells

    def _get_neighbors(self, cells):
        neighbors = {}
        for i, code in enumerate(cells):
            print "\rGetting neighbors for cell:", (i+1),
            neighbors[code] = set()
            for other_code in cells:
                d = self.cell_dist_measure(code, other_code)
                if d <= self.overlapping_radius:
                    neighbors[code].add(other_code)
        print ""
        return neighbors

    @staticmethod
    def _get_non_redundant_cells(cells, neighbors):
        non_redundant = set(cells)
        for code, other_code in product(cells, cells):
            if code != other_code and other_code in non_redundant and neighbors[other_code] <= neighbors[code]:
                non_redundant.remove(other_code)
        return non_redundant

    @staticmethod
    def _overlap_cells(non_redundant, cells, neighbors):
        expanded_cells = {}
        for cell in non_redundant:
            dataset = ([], [])
            for neighbor in neighbors[cell]:
                dataset[0].extend(cells[neighbor][0])
                dataset[1].extend(cells[neighbor][1])
            expanded_cells[cell] = dataset
        return expanded_cells
