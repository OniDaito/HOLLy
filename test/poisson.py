""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

poisson.py - test the poisson point sampler
"""

import unittest
from util.poisson import PoissonSampler


class Poisson(unittest.TestCase):
    def test_dist(self):
        sampler = PoissonSampler(1000)
        sample = sampler.sample(100)
        print(sample[0])
     
