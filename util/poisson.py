"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

poisson.py - a poisson sampler for points in the
initial setup

"""

import math
from typing import Tuple
import torch
import os
import random
import pickle
import array
import numpy as np
from tqdm import tqdm
from numba.typed import List
import heapq
from numba import jit
from pyquaternion import Quaternion
from data.loader import Loader
from util.math import Points, Point, PointsTen


def gen_weight(p, samples, rmax):
    w = 0
    for j in samples:
        d = dist(p, j)
        if d < 2.0 * rmax:
            w += math.pow(1 - (d / (2.0 * rmax)), 8)
    return 1 / w  # instead of w as python uses min heaps


def redo_heap(heap: list, points: list, indices: list, new_sample: List, rmax: float):
    new_heap = []

    for hidx in range(len(heap)):
        (w, i) = heap[hidx]
        if i in indices:
            w = gen_weight(points[i], new_sample, rmax)

        heapq.heappush(new_heap, (w, i))

    return new_heap


def dist(p: Tuple, q: Tuple):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2)


class PoissonSampler(object):
    # http://www.cemyuksel.com/cyCodeBase/soln/poisson_disk_sampling.html
    # Optionally specify an existing set of points.
    def __init__(self, num_points):
        from sklearn.neighbors import BallTree
        self.points = []
        self.num_points = num_points
      
        for i in range(num_points):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(-1.0, 1.0)
            self.points.append((x, y, z))

        self.points = np.array(self.points)
        self.tree = BallTree(self.points, metric=dist)

    def sample(self, sample_size):
        # TODO - as heapq is a lowest weight first and we end up deleting and
        # recreating, there might be a better way?
        heap = []
        # 1.0 here is the A3 or volume of the sample space
        self.rmax = (1.0 / (4.0 * math.sqrt(sample_size))) ** (1.0 / 3.0)
        # initial weights
        for idx in range(len(self.points)):
            w = gen_weight(self.points[idx], self.points, self.rmax)
            heapq.heappush(heap, (w, idx))

        new_sample = self.points.copy()
        removals = []

        # Go through until our sample is reduced to the sample
        # size requested.
        #  with tqdm(total=self.num_points-sample_size) as pbar:
        while len(new_sample) > sample_size:
            (q, idx) = heapq.heappop(heap)
            removals.append(idx)
            indices = self.tree.query_radius(
                self.points[idx].reshape(1, -1), r=self.rmax*2.0)
            indices = list(indices[0])
            new_sample = List()
            for i in range(len(self.points)):
                if i not in removals:
                    new_sample.append(self.points[i])

            #  pbar.update(1)
            heap = redo_heap(heap, self.points, indices, new_sample, self.rmax)

        return new_sample


