""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

renderer.py - our test functions using python's unittest.
This file tests the renderer

"""
import unittest

import torch
import random
import math
import util.plyobj as plyobj
from net.renderer import Splat
from util.image import save_image
from util.math import TransTen, PointsTen, VecRot


class Render(unittest.TestCase):

    def test_render(self):
        use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
        base_points = PointsTen(device=device)
        base_points.from_points(plyobj.load_obj("./objs/bunny_large.obj"))

        mask = []
        for _ in range(len(base_points)):
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        xt = torch.tensor([1.0], dtype=torch.float32)
        yt = torch.tensor([0.2], dtype=torch.float32)

        splat = Splat(size=(128, 256), device=device)

        r = VecRot(0, 0, 0).to_ten(device=device)
        r.random()
        t = TransTen(xt, yt)

        model = splat.render(base_points, r, t, mask, sigma=1.8)
        self.assertTrue(torch.sum(model) > 300)
        save_image(model, name="test_renderer_0.jpg")

    def test_tall(self):
        use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
        base_points = PointsTen(device=device)
        base_points.from_points(plyobj.load_obj("./objs/bunny_large.obj"))

        mask = []
        for _ in range(len(base_points)):
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        xt = torch.tensor([0.0], dtype=torch.float32)
        yt = torch.tensor([0.0], dtype=torch.float32)

        splat = Splat(size=(256, 128), device=device)

        r = VecRot(0, 0, 0).to_ten(device=device)
        t = TransTen(xt, yt)

        model = splat.render(base_points, r, t, mask, sigma=1.8)
        self.assertTrue(torch.sum(model) > 200)
        save_image(model, name="test_renderer_tall.jpg")

    def test_dropout(self):
        use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
        base_points = PointsTen(device=device)
        base_points.from_points(plyobj.load_obj("./objs/bunny_large.obj"))
        mask = []

        for _ in range(len(base_points)):
            if random.uniform(0, 1) >= 0.5:
                mask.append(1.0)
            else:
                mask.append(0.0)

        mask = torch.tensor(mask, device=device)
        xt = torch.tensor([0.9], dtype=torch.float32)
        yt = torch.tensor([0.1], dtype=torch.float32)

        splat = Splat(device=device)
        r = VecRot(0, math.radians(90), 0).to_ten(device=device)
        t = TransTen(xt, yt)

        model = splat.render(base_points, r, t, mask, sigma=1.8)

        mask = []

        for _ in range(len(base_points)):
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        model2 = splat.render(base_points, r, t, mask, sigma=1.8)

        self.assertTrue(torch.sum(model2) > torch.sum(model))
        save_image(model, name="test_renderer_1.jpg")


if __name__ == "__main__":
    unittest.main()
