""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

math.py - testing some of the maths functions from
util/math.py
"""

import unittest
import math
from random import random
from util.math import PointsTen, gen_mat_from_rod, mat_to_rod, VecRot, Point, Points


class Math(unittest.TestCase):
    def test_gen_mat(self):
        a = VecRot(math.radians(90), 0, 0).to_ten()
        m = gen_mat_from_rod(a)
        (u, b) = mat_to_rod(m)
        self.assertTrue(math.fabs(b - math.radians(90)) < 0.01)

    def test_vec_rot(self):
        a = VecRot(math.radians(90), 0, 0)
        self.assertTrue(math.fabs(a.get_length() - math.radians(90)) < 0.01)

        nl = a.get_normalised()
        self.assertTrue(math.fabs(nl[0] - 1.0) < 0.0001)

    def test_point_rot(self):
        points = Points()
        r = VecRot(0, 0, math.radians(90))

        for i in range(10):
            points.append(Point(i * 0.1, 0, 0))

        rot_points = r.rotate_points(points)
        self.assertTrue(math.fabs(rot_points[9].y - 0.9) < 0.0001)

    def test_vec_loss(self):
        from train import calculate_move_loss

        points = Points()
        points2 = Points()

        for i in range(10):
            points.append(Point(i * 0.1 + 0.1, 0, 0))
            points2.append(Point(i * 0.1 + 0.4, 0, 0))

        tp = PointsTen().from_points(points)
        tp2 = PointsTen().from_points(points2)
        loss = calculate_move_loss(tp, tp2)
        print("Move Loss", loss)
        points = Points()
        points2 = Points()

        for i in range(10):
            points.append(Point(random(), random(), random(), 1.0))
            points2.append(Point(random(), random(), random(), 1.0))

        tp = PointsTen().from_points(points)
        tp2 = PointsTen().from_points(points2)
        loss = calculate_move_loss(tp, tp2)
        print("Move Loss", loss)
