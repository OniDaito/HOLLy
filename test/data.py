""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

data.py - our test functions using python's unittest.
This file tests the data classes like loader and such.

"""
import unittest
import math
import torch
import os
import random
import torch.nn.functional as F
from data.loader import Loader
from data.imageload import ImageLoader
from data.sets import DataSet, SetType
from data.buffer import Buffer, BufferImage
from data.batcher import Batcher
from net.renderer import Splat
from util.math import vec_to_quat, qdist
from util.render import render
from util.image import NormaliseBasic


class Data(unittest.TestCase):
    def test_loader(self):
        """Perform a series of tests on our DataLoader class. Eventually, we shall
        move this to a proper test suite."""
        splat = Splat(device="cpu")

        with torch.no_grad():
            d = Loader(size=100, objpath="./objs/teapot_large.obj",
                       wobble=0.0, max_spawn=1)

            d.set_sigma(sigma=4.0)

            # for p, m, r, t, sig in [t.unpack() for t in d]:
            #     print(p, m, r, t, sig)
            #     break

            (p, m, r, t, sig) = d[50].unpack()
            out = render(p, m, r, t, sig, splat)

            # We should have something rendered with intensity > 200
            self.assertTrue(torch.sum(out) > 200)

            # save_image(out.cpu().detach().numpy(), "dataload_test_0.jpg")
            # save_fits(out.cpu().detach().numpy(), "dataload_test_0.fits")

    def test_set(self):
        """ Test the set class that lives above the loader."""
        splat = Splat(device="cpu")
        # Initial setup of PyTorch
        loader = Loader(size=96, objpath="./objs/bunny_large.obj", wobble=0.0)

        loader.set_sigma(2.0)
        device = torch.device("cpu")
        dataset = DataSet(SetType.TRAIN, 96, loader, device)

        # for idx, v in enumerate(dataset):
        #    nval, mask, rots, sigs = v.unpack()
        #    print(idx, rots)

        (p, m, r, t, sig) = dataset[0].unpack()
        out0 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out0) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_0a.jpg")

        (p, m, r, t, sig) = dataset[32].unpack()
        out1 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out1) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_0b.jpg")

        (p, m, r, t, sig) = dataset[64].unpack()
        out2 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out2) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_0c.jpg")

        dataset.shuffle_chunk(size=32)

        (p, m, r, t, sig) = dataset[0].unpack()
        out3 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out3) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_1a.jpg")

        (p, m, r, t, sig) = dataset[32].unpack()
        out4 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out4) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_1b.jpg")

        (p, m, r, t, sig) = dataset[64].unpack()
        out5 = render(p, m, r, t, sig, splat)
        self.assertTrue(torch.sum(out5) > 300)
        # save_image(out.cpu().detach().numpy(), "dataload_test_1c.jpg")

    def test_buffer(self):
        """ Test our buffer class that sits above the set."""

        splat = Splat(device="cpu")
        loader = Loader(size=200, objpath="./objs/teapot_large.obj")

        loader2 = Loader(size=200, objpath="./objs/teapot_large.obj")

        device = torch.device("cpu")

        dataset = DataSet(SetType.TRAIN, 200, loader, deterministic=True)
        dataset2 = DataSet(SetType.TRAIN, 200, loader2, deterministic=True)

        buffer = Buffer(dataset, splat, buffer_size=100, device=device)
        buffer2 = Buffer(dataset2, splat, buffer_size=50, device=device)

        dataset.reset()
        buffer.fill()
        dataset2.reset()
        buffer2.fill()

        out0 = buffer[0].flatten()[0]
        # save_image(out.cpu().detach().numpy(), "databuffer_test_0a.jpg")

        (p, m, r, t, sig) = dataset[0].unpack()
        out1 = render(p, m, r, t, sig, splat)

        # save_image(out.cpu().detach().numpy(), "databuffer_test_0b.jpg")
        self.assertTrue(torch.sum(torch.abs(torch.sub(out0, out1))) < 1.0)

        for i in range(90):
            buffer.__next__()
        (datum, r, t, sigma) = buffer.__next__().flatten()
        self.assertFalse(torch.sum(torch.abs(torch.sub(datum, out1))) < 1.0)
        # save_image(datum.cpu().detach().numpy(), "databuffer_test_0c.jpg")

        out2 = buffer2[0][0]
        # save_image(out.cpu().detach().numpy(), "databuffer_test_1a.jpg")

        (p, m, r, t, sig) = dataset2[90].unpack()
        out3 = render(p, m, r, t, sig, splat)
        # save_image(out.cpu().detach().numpy(), "databuffer_test_1b.jpg")
        self.assertFalse(torch.sum(torch.abs(torch.sub(out2, out3))) < 1.0)

        for i in range(90):
            buffer2.__next__()
        (datum, r, t, sigma) = buffer2.__next__()
        # save_image(datum.cpu().detach().numpy(), "databuffer_test_1c.jpg")
        self.assertTrue(torch.sum(torch.abs(torch.sub(datum, out3))) < 1.0)

    def test_batcher(self):
        """ Test the batcher."""
        splat = Splat(device="cpu")
        loader = Loader(size=200, objpath="./objs/teapot_large.obj")

        dataset = DataSet(SetType.TRAIN, 200, loader, deterministic=True)

        buffer = Buffer(dataset, splat, buffer_size=100, device="cpu")

        batcher = Batcher(buffer)

        for i, b in enumerate(batcher):
            self.assertTrue(len(b.rotations) == 3)
            self.assertTrue(len(b.translations) == 2)

    def test_normalise(self):
        """ Test the normaliser."""
        splat = Splat(device="cpu")
        loader = Loader(size=200, objpath="./objs/teapot_large.obj")
        dataset = DataSet(SetType.TRAIN, 200, loader, deterministic=True)
        buffer = Buffer(dataset, splat, buffer_size=100, device="cpu")
        batcher = Batcher(buffer)
        normaliser = NormaliseBasic()

        for i, b in enumerate(batcher):
            target = b.datum
            self.assertTrue(torch.sum(target[0]) != 100.0)
            normalised = normaliser.normalise(target.reshape(16, 1, 128, 128))
            self.assertTrue(torch.sum(normalised[0]) == 100.0)
            break

    def test_wobble(self):
        splat = Splat(device="cpu")

        random.seed(30)
        d0 = Loader(
            size=100, objpath="./objs/teapot_large.obj", wobble=0.001)

        random.seed(30)
        d1 = Loader(size=100, objpath="./objs/teapot_large.obj", wobble=0.06)

        d1.set_sigma(sigma=2.0)

        random.seed(30)
        d2 = Loader(size=100, objpath="./objs/teapot_large.obj", wobble=0.124)

        d2.set_sigma(sigma=2.0)

        (p, m, r, t, sig) = d0[50].unpack()
        out0 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d1[50].unpack()
        out1 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d2[50].unpack()
        out2 = render(p, m, r, t, sig, splat)

        self.assertFalse(torch.sum(torch.abs(torch.sub(out1, out0))) < 1.0)
        self.assertFalse(torch.sum(torch.abs(torch.sub(out2, out1))) < 1.0)

        #from util.image import save_image
        # save_image(out0.cpu().detach().numpy(),
        #            "dataload_wobble_test_0.jpg")
        # save_image(out1.cpu().detach().numpy(),
        #            "dataload_wobble_test_1.jpg")

    def test_spawn(self):
        splat = Splat(device="cpu")

        with torch.no_grad():
            random.seed(42)
            d = Loader(
                size=100, objpath="./objs/bunny_large.obj", wobble=0.05, max_spawn=3
            )
            d.set_sigma(sigma=2.0)

            random.seed(42)
            d2 = Loader(
                size=100, objpath="./objs/bunny_large.obj", wobble=0.05, max_spawn=3
            )
            d2.set_sigma(sigma=2.0)

            random.seed(42)
            d3 = Loader(
                size=100, objpath="./objs/bunny_large.obj", wobble=0.0, max_spawn=1
            )
            d3.set_sigma(sigma=2.0)

            (p, m, r, t, sig) = d[50].unpack()
            out0 = render(p, m, r, t, sig, splat)

            (p, m, r, t, sig) = d2[50].unpack()
            out1 = render(p, m, r, t, sig, splat)

            (p, m, r, t, sig) = d3[50].unpack()
            out2 = render(p, m, r, t, sig, splat)

            self.assertTrue(torch.sum(torch.abs(torch.sub(out0, out1))) < 1.0)
            self.assertFalse(torch.sum(torch.abs(torch.sub(out0, out2))) < 1.0)

            # save_image(out.cpu().detach().numpy(),
            #           "dataload_spawn_test_0.jpg")

    def test_augment(self):
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        splat = Splat(device="cpu")
        d0 = Loader(
            size=2, objpath="./objs/teapot_large.obj", dropout=0.0,
            augment=True, num_augment=10)

        d0.set_sigma(sigma=2.0)

        (p, m, r, t, sig) = d0[0].unpack()
        out0 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d0[1].unpack()
        out1 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d0[2].unpack()
        out2 = render(p, m, r, t, sig, splat)

        self.assertTrue(d0.size == 20)
        #self.assertFalse(torch.sum(torch.abs(torch.sub(out1, out0))) < 0.1)

        from util.image import save_image
        save_image(out0.cpu().detach().numpy(),
                   "dataload_augment_test_0.jpg")
        save_image(out1.cpu().detach().numpy(),
                   "dataload_augment_test_1.jpg")
        save_image(out2.cpu().detach().numpy(),
                   "dataload_augment_test_2.jpg")

        # Take a look at the difference this makes on quat distances
        data_norm = Loader(
            size=40000, objpath="./objs/teapot_large.obj")

        data_aug = Loader(
            size=4000, objpath="./objs/teapot_large.obj", augment=False, num_augment=10)

        # Plot heatmaps
        # Take each set of differences and turn them into a histogram
        qbase = [0, 0, 0, 1.0]
        num_classes = 100
        bin_size = math.sqrt(2) / num_classes
        bins = [i * bin_size for i in range(int(math.sqrt(2)/bin_size))]

        hists = []
        diffs = []

        for d in data_norm:
            (p, m, r, t, sig) = d.unpack()
            qnew = vec_to_quat(r)
            diffs.append(qdist(qbase, qnew))

        hist, edges = np.histogram(diffs, bins=bins)
        hists.append(hist)

        dframe = pd.DataFrame(hists)
        ax = sns.heatmap(data=dframe)
        ax.set(title="Rotation Different Heatmap for a normal loader.",
            xlabel="Difference in 100 Bins.",
            ylabel="Steps in training.",)
        plt.show()

        hists = []
        diffs = []
        
        for d in data_aug:
            (p, m, r, t, sig) = d.unpack()
            qnew = vec_to_quat(r)
            diffs.append(qdist(qbase, qnew))

        hist, edges = np.histogram(diffs, bins=bins)
        hists.append(hist)

        dframe = pd.DataFrame(hists)
        ax = sns.heatmap(data=dframe)
        ax.set(title="Rotation Different Heatmap for an augmented loader.",
            xlabel="Difference in 100 Bins.",
            ylabel="Steps in training.",)

        # Create our heatmap from the histograms with Seaborn and Pandas.
        dframe = pd.DataFrame(hists)
        ax = sns.heatmap(data=dframe)
        ax.set(title="Rotation Different Heatmap.",
            xlabel="Difference in 100 Bins.",
            ylabel="Steps in training.",)
        plt.show()


    def test_all(self):
        splat = Splat(device="cpu")

        with torch.no_grad():
            random.seed(42)
            d = Loader(
                size=100,
                objpath="./objs/bunny_large.obj",
                wobble=0.05,
                max_spawn=6,
                dropout=0.5,
            )
            d.set_sigma(sigma=1.2)

            random.seed(42)
            d2 = Loader(size=100, objpath="./objs/bunny_large.obj")
            d2.set_sigma(sigma=1.2)

            (p, m, r, t, sig) = d[50].unpack()
            out0 = render(p, m, r, t, sig, splat)

            (p, m, r, t, sig) = d2[50].unpack()
            out1 = render(p, m, r, t, sig, splat)

            self.assertFalse(torch.sum(torch.abs(torch.sub(out0, out1))) < 1.0)

            # save_image(out.cpu().detach().numpy(),
            #           "dataload_all_test_0.jpg")

    def test_imageloader(self):
        """ Test the image loader with some FITS images."""

        loader = ImageLoader(size=10, image_path="./test/images/", sigma=2.0)
        self.assertTrue(len(loader) == 10)
        out = loader[5].unpack()

        dataset = DataSet(SetType.TRAIN, 10, loader)
        buffer = BufferImage(dataset, buffer_size=10, device="cpu")
        batcher = Batcher(buffer, batch_size=2)

        for b in batcher:
            self.assertTrue(len(b.data) == 2)

        buffer.fill()
        out = buffer[0]
        self.assertTrue(out.datum.shape[0] == 128)

    def test_dropout(self):
        """Perform a series of tests on our DataLoader class. Eventually, we shall
        move this to a proper test suite."""
        from util.image import save_image
        splat = Splat(device="cpu")

        with torch.no_grad():
            d = Loader(size=10, objpath="./objs/bunny_large.obj", wobble=0.0,
                       max_spawn=1, dropout=0.8, translate=False, rotate=False)

            d.set_sigma(sigma=2.0)
            _, pm, _, _, _ = d[0].unpack()

            for i, d in enumerate(d):
                if i != 0:
                    p, m, r, t, sig = d.unpack()
                    self.assertFalse(torch.equal(pm.to_ten(), m.to_ten()))
                    out = render(p, m, r, t, sig, splat)
                    save_image(out.cpu().detach().numpy(),
                               "dataload_test_" + str(i) + ".jpg")
                    pm = m

    def test_load_save(self):
        splat = Splat(device="cpu")
        loader = Loader(size=200, objpath="./objs/teapot_large.obj")
        dataset = DataSet(SetType.TRAIN, 100, loader, deterministic=True)
        dataset_test = DataSet(SetType.TEST, 100, loader, deterministic=True)
        buffer = Buffer(dataset, splat, buffer_size=100, device="cpu")
        batcher = Batcher(buffer)
        target = batcher.__next__().data[0]

        loader.save("test_loader.pickle")
        dataset.save("test_dataset_train.pickle")
        dataset_test.save("test_dataset_test.pickle")

        loader2 = Loader(size=200, objpath="./objs/teapot_large.obj")
        loader2.load("test_loader.pickle")
        dataset2 = DataSet(SetType.TRAIN, 100, loader2, deterministic=True)
        dataset2.load("test_dataset_train.pickle")
        buffer2 = Buffer(dataset2, splat, buffer_size=100, device="cpu")
        batcher2 = Batcher(buffer2)
        target2 = batcher2.__next__().data[0]

        self.assertTrue(torch.equal(target, target2))

        try:
            os.remove("test_loader.pickle")
            os.remove("test_dataset_train.pickle")
            os.remove("test_dataset_test.pickle")
        except OSError:
            print ("OSError")
            pass

    def test_losses(self):
        splat = Splat(device="cpu")
        d = Loader(size=10, objpath="./objs/bunny_large.obj")

        d.set_sigma(sigma=2.0)
        p, m, r, t, sig = d[0].unpack()
        base0 = render(p, m, r, t, sig, splat)
        p, m, r, t, sig = d[1].unpack()
        base1 = render(p, m, r, t, sig, splat)

        out0 = base0.reshape(1, 1, 128, 128)
        out1 = base1.reshape(1, 1, 128, 128)

        norm_core = NormaliseBasic()
        norm_large = NormaliseBasic()
        norm_large.factor = 1000.0

        print("Intensities")
        print(torch.sum(out0), torch.sum(out1))

        ncore0 = norm_core.normalise(out0)
        ncore1 = norm_core.normalise(out1)

        print("Norm core Intensities")
        print(torch.sum(ncore0), torch.sum(ncore1))
        
        nbig0 = norm_large.normalise(out0)
        nbig1 = norm_large.normalise(out1)

        print("Norm large Intensities")
        print(torch.sum(nbig0), torch.sum(nbig1))

        print("Pre normed Single image losses")
        print("Loss smol:", F.l1_loss(out0, out1, reduction="sum").item())
        print("Loss big:", F.l1_loss(out0, out1, reduction="mean").item())

        print("Single image losses")
        print("Loss smol:", F.l1_loss(ncore0, ncore1, reduction="sum").item())
        print("Loss big:", F.l1_loss(nbig0, nbig1, reduction="mean").item())

        bcore0 = base0.reshape(1, 128, 128)
        bcore0 = bcore0.repeat(32, 1, 1, 1)
        bcore1 = base1.reshape(1, 128, 128)
        bcore1 = bcore1.repeat(32, 1, 1, 1)
        bcore0 = norm_core.normalise(bcore0)
        bcore1 = norm_core.normalise(bcore1)

        bbig0 = base0.reshape(1, 128, 128)
        bbig0 = bbig0.repeat(32, 1, 1, 1)
        bbig1 = base1.reshape(1, 128, 128)
        bbig1 = bbig1.repeat(32, 1, 1, 1)
        bbig0 = norm_large.normalise(bbig0)
        bbig1 = norm_large.normalise(bbig1)

        print("Batch size 32 image losses")
        print("Loss smol:", F.l1_loss(bcore0, bcore1, reduction="sum").item())
        print("Loss big:", F.l1_loss(bbig0, bbig1, reduction="mean").item())

    def test_paper(self):
        splat = Splat(device="cpu")

        random.seed(30)
        d0 = Loader(
            size=10, objpath="./objs/bunny_large.obj", wobble=0.03, rotate=False)
        random.seed(30)

        d1 = Loader(
            size=10, objpath="./objs/bunny_large.obj", wobble=0.06, rotate=False)
        random.seed(30)

        d2 = Loader(
            size=10, objpath="./objs/bunny_large.obj", wobble=0.09, rotate=False)
        random.seed(30)

        d3 = Loader(
            size=10, objpath="./objs/bunny_large.obj", wobble=0.12, rotate=False)
        random.seed(30)

        d4 = Loader(
            size=10, objpath="./objs/bunny_large.obj", wobble=0.15, rotate=False)

        d0.set_sigma(sigma=2.0)
        d1.set_sigma(sigma=2.0)
        d2.set_sigma(sigma=2.0)
        d3.set_sigma(sigma=2.0)
        d4.set_sigma(sigma=2.0)

        (p, m, r, t, sig) = d0[0].unpack()
        out0 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d1[0].unpack()
        out1 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d2[0].unpack()
        out2 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d3[0].unpack()
        out3 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d4[0].unpack()
        out4 = render(p, m, r, t, sig, splat)

        from util.image import save_image
        save_image(out0.cpu().detach().numpy(),
                   "dataload_paper_wobble_0.jpg")
        save_image(out1.cpu().detach().numpy(),
            "dataload_paper_wobble_1.jpg")
        save_image(out2.cpu().detach().numpy(),
            "dataload_paper_wobble_2.jpg")
        save_image(out3.cpu().detach().numpy(),
            "dataload_paper_wobble_3.jpg")
        save_image(out4.cpu().detach().numpy(),
            "dataload_paper_wobble_4.jpg")

        random.seed(30)
        d0 = Loader(
            size=10, objpath="./objs/bunny_large.obj", dropout=0.1, rotate=False)
        random.seed(30)

        d1 = Loader(
            size=10, objpath="./objs/bunny_large.obj", dropout=0.3, rotate=False)
        random.seed(30)

        d2 = Loader(
            size=10, objpath="./objs/bunny_large.obj", dropout=0.5, rotate=False)
        random.seed(30)

        d3 = Loader(
            size=10, objpath="./objs/bunny_large.obj", dropout=0.7, rotate=False)
        random.seed(30)

        d4 = Loader(
            size=10, objpath="./objs/bunny_large.obj", dropout=0.9, rotate=False)

        d0.set_sigma(sigma=2.0)
        d1.set_sigma(sigma=2.0)
        d2.set_sigma(sigma=2.0)
        d3.set_sigma(sigma=2.0)
        d4.set_sigma(sigma=2.0)

        (p, m, r, t, sig) = d0[0].unpack()
        out0 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d1[0].unpack()
        out1 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d2[0].unpack()
        out2 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d3[0].unpack()
        out3 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d4[0].unpack()
        out4 = render(p, m, r, t, sig, splat)

        save_image(out0.cpu().detach().numpy(),
                   "dataload_paper_drop_0.jpg")
        save_image(out1.cpu().detach().numpy(),
            "dataload_paper_drop_1.jpg")
        save_image(out2.cpu().detach().numpy(),
            "dataload_paper_drop_2.jpg")
        save_image(out3.cpu().detach().numpy(),
            "dataload_paper_drop_3.jpg")
        save_image(out4.cpu().detach().numpy(),
            "dataload_paper_drop_4.jpg")

        random.seed(30)
        d0 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=4, spawn=0.3, wobble=0.06, rotate=False)
        random.seed(30)

        d1 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=4, spawn=0.3, wobble=0.12, rotate=False)
        random.seed(30)

        d2 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=4, spawn=0.7, wobble=0.06, rotate=False)
        random.seed(30)

        d3 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=4, spawn=0.7, wobble=0.12, rotate=False)
        random.seed(30)

        d4 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=8,  spawn=0.3, wobble=0.06, rotate=False)

        d5 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=8,  spawn=0.3, wobble=0.12, rotate=False)

        d6 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=8,  spawn=0.7, wobble=0.06, rotate=False)

        d7 = Loader(
            size=10, objpath="./objs/bunny_large.obj", max_spawn=8,  spawn=0.7, wobble=0.12, rotate=False)

        d0.set_sigma(sigma=2.0)
        d1.set_sigma(sigma=2.0)
        d2.set_sigma(sigma=2.0)
        d3.set_sigma(sigma=2.0)
        d4.set_sigma(sigma=2.0)
        d5.set_sigma(sigma=2.0)
        d6.set_sigma(sigma=2.0)
        d7.set_sigma(sigma=2.0)

        (p, m, r, t, sig) = d0[0].unpack()
        out0 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d1[0].unpack()
        out1 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d2[0].unpack()
        out2 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d3[0].unpack()
        out3 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d4[0].unpack()
        out4 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d5[0].unpack()
        out5 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d6[0].unpack()
        out6 = render(p, m, r, t, sig, splat)

        (p, m, r, t, sig) = d7[0].unpack()
        out7 = render(p, m, r, t, sig, splat)

        save_image(out0.cpu().detach().numpy(),
                   "dataload_paper_ns_0.jpg")
        save_image(out1.cpu().detach().numpy(),
            "dataload_paper_ns_1.jpg")
        save_image(out2.cpu().detach().numpy(),
            "dataload_paper_ns_2.jpg")
        save_image(out3.cpu().detach().numpy(),
            "dataload_paper_ns_3.jpg")
        save_image(out4.cpu().detach().numpy(),
            "dataload_paper_ns_4.jpg")
        save_image(out5.cpu().detach().numpy(),
            "dataload_paper_ns_5.jpg")
        save_image(out6.cpu().detach().numpy(),
            "dataload_paper_ns_6.jpg")
        save_image(out7.cpu().detach().numpy(),
            "dataload_paper_ns_7.jpg")

if __name__ == "__main__":
    unittest.main()
