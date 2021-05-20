""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

mesh.py

A small test that checks the numbers given by the mesh_score
program. This program is external and not included by 
default so this test is not included in the suite.

"""

import unittest

from util.points import init_points
from util.plyobj import save_ply

from subprocess import Popen, PIPE


class Mesh(unittest.TestCase):

    def test_random(self):
        points_a = init_points(num_points=250, device="cpu")
        points_b = init_points(num_points=250, device="cpu")

        points_a = points_a.get_points()
        points_b = points_b.get_points()

        save_ply("points_a.ply", points_a.get_iterable())
        save_ply("points_b.ply", points_b.get_iterable())

        process = Popen(["mesh_score", "-b", "points_a.ply", "-t", "points_b.ply"], stdout=PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
        print(output, exit_code)
