""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train.py - testing some of the functions in the
training file as well as the Net class.

"""

import unittest
import math
from train import cont_sigma
from util.math import PointsTen


class Args:
    def __init__(self):
        self.train_size = 240000
        self.epochs = 3
        self.batch_size = 32
        self.aug = False


class Train(unittest.TestCase):
    def test_cont(self):
        sigma_lookup = [
            10,
            9.0,
            8.1,
            7.29,
            6.56,
            5.9,
            5.31,
            4.78,
            4.3,
            3.87,
            3.65,
            3.28,
            2.95,
            2.66,
            2.39,
            2.15,
            1.94,
            1.743,
            1.57,
            1.41,
        ]
        sigma = 10
        args = Args()
        sigmas = []
        args.epochs = 10
        tb = int(240000 / 32)

        for epoch in range(3):
            for i in range(tb):
                sigma = cont_sigma(args, epoch, i, tb, sigma_lookup)
            sigmas.append(sigma)

        print(sigmas)

        self.assertTrue(math.fabs(sigmas[0] - 8.2) < 0.1)
        self.assertTrue(math.fabs(sigmas[1] - 6.7) < 0.1)
        self.assertTrue(math.fabs(sigmas[2] - 5.4) < 0.1)

        args.epochs = 20
        sigmas = []
        sigma = 10
        sigma_lookup = [10, 6.0, 8.0, 7.0, 6.0, 5.0, 6.0, 8.0, 6.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0]

        for epoch in range(args.epochs):
            for i in range(tb):
                sigma = cont_sigma(args, epoch, i, tb, sigma_lookup)
            sigmas.append(sigma)

        print(sigmas)

    def test_draw_sigma(self):
        import seaborn as sns

        sigma_lookup = [
            10,
            10,
            10,
            10,
            9.0,
            9.0,
            9.0,
            9.0,
            8.1,
            8.1,
            8.1,
            8.1,
            7.29,
            7.29,
            7.29,
            7.29,
            6.56,
            6.56,
            6.56,
            6.56,
            5.9,
            5.9,
            5.9,
            5.9,
            5.31,
            5.31,
            5.31,
            5.31,
            4.78,
            4.78,
            4.78,
            4.78,
            4.3,
            4.3,
            3.87,
            3.87,
            3.65,
            3.65,
            3.28,
            3.28,
        ]
        args = Args()
        args.epochs = 40
        args.train_size = 20000
        sigmas = [10]
        sigma = 10
        for epoch in range(args.epochs):
            sigma = sigma_lookup[epoch]
            for i in range(int(args.train_size / args.batch_size)):
                # sigma = cont_sigma(args, epoch, sigma, sigma_lookup)

                sigmas.append(sigma)

        sns.set_theme()
        sns_plot = sns.relplot(
            data=sigmas,
            kind="line",
            facet_kws=dict(sharex=False),
        )
        sns_plot.set(
            xlabel="Epoch", ylabel="Sigma", title="Sigma value for image rendering."
        )
        sns_plot.set(ylim=(0, 12))
        sns_plot.savefig("output.png")

    def test_draw_graph(self):
        import torch.autograd
        import random
        import torch.nn.functional as F
        from util.plyobj import load_obj
        import math
        from net.net import Net, draw_graph
        from net.renderer import Splat
        from util.math import VecRotTen, TransTen
        from rich import print

        device = torch.device("cpu")

        xrb = random.random() * math.pi
        yrb = random.random() * math.pi
        zrb = random.random() * math.pi
        xtb = random.random() * 2.0 - 1.0
        ytb = random.random() * 2.0 - 1.0

        xr = torch.tensor([xrb], dtype=torch.float32, device=device)
        yr = torch.tensor([yrb], dtype=torch.float32, device=device)
        zr = torch.tensor([zrb], dtype=torch.float32, device=device)
        xt = torch.tensor([xtb], dtype=torch.float32, device=device)
        yt = torch.tensor([ytb], dtype=torch.float32, device=device)

        r = VecRotTen(xr, yr, zr)
        t = TransTen(xt, yt)

        # Setup our splatting pipeline which is added to both dataloader
        # and our network as they use thTraine same settings
        splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)
        model = Net(splat)
        model.to(device)
        loaded_points = load_obj(objpath="./objs/teapot.obj")
        mask = torch.ones((len(loaded_points)), dtype=torch.float32)
        batch_size = 1

        with torch.no_grad():
            mask = torch.tensor(mask, device=device)

            # m = scale_cloud(loaded_points)
            # TODO - Don't bother with a stretch on this eval for now
            loaded_points = PointsTen().from_points(loaded_points)
            result = splat.render(loaded_points, r, t, mask)
            target = result.reshape(batch_size, 128, 128)
            target = target.repeat(1, 1, 1, 1)
            target = target.to(device)
            model.set_sigma(2.0)

        loaded_points.data.requires_grad_(requires_grad=True)
        output = model.forward(target, loaded_points.data)
        output = output.reshape(1, 1, 128, 128)
        output = output.to(device)
        loss = F.l1_loss(output, target)
        print("Loss:", loss)

        # run the new command using the given tracer
        # torch.autograd.backward(loss, retain_graph=True)
        loss.backward(create_graph=True)

        # print(model.parameters)
        # print(dir(model))
        # print(model.state_dict)

        watching = [
            ("Points", loaded_points.data),
            ("fc2", model.fc2.weight),
            ("fc1", model.fc1.weight),
            ("conv1", model.conv1.weight),
            ("conv2", model.conv2.weight),
            ("conv3", model.conv3.weight),
            ("conv4", model.conv4.weight),
            ("conv5", model.conv5.weight),
            ("conv6", model.conv6.weight),
        ]

        # for idx, off in enumerate(offset_stack):
        #    watching.append(("Offsets[[" + str(idx) + "]]", off))

        draw_graph(loss, watching)

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    unittest.main()
