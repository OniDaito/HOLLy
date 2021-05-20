""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

test.py - our test functions using python's unittest


"""

import unittest
from test.data import Data
from test.render import Renderer
from test.train import Train
from test.math import Math

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Data("test_loader"))
    suite.addTest(Data("test_set"))
    suite.addTest(Data("test_buffer"))
    suite.addTest(Data("test_batcher"))
    suite.addTest(Data("test_wobble"))
    suite.addTest(Data("test_spawn"))
    suite.addTest(Data("test_all"))
    suite.addTest(Data("test_imageloader"))

    suite.addTest(Renderer("test_render"))
    suite.addTest(Renderer("test_dropout"))

    suite.addTest(Train("test_cont"))
    suite.addTest(Train("test_draw_graph"))

    suite.addTest(Math("test_gen_mat"))
    suite.addTest(Math("test_vec_rot"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
    unittest.main()
