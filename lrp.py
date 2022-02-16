""" 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/  
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

lrp_shaper.py - An implementation of the Layer-wise 
relevance propagation (LRP) algorithm to see what a 
neural network is actually doing. 

Based on the work from: 
    https://github.com/sebastian-lapuschkin/lrp_toolbox/
    http://jmlr.org/papers/volume17/15-618/15-618.pdf
    http://www.heatmapping.org/tutorial/

"""


import copy
import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import os
from net.net import num_flat_features
from util.loadsave import load_checkpoint, load_model
from util.plyobj import load_obj
from util.math import PointsTen, VecRotTen, TransTen
from util.image import save_image, NormaliseBasic, NormaliseNull
from net.renderer import Splat
import matplotlib

matplotlib.use("Agg")


def toconv(layer):
    if isinstance(layer, nn.Linear):
        newlayer = None
        m, n = layer.weight.shape[1], layer.weight.shape[0]
        newlayer = nn.Conv2d(m, n, 1)
        newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
        newlayer.bias = nn.Parameter(layer.bias)

        return newlayer

    return layer


def heatmap(R, sx, sy, filename):
    print("heatmap shapes", R.shape, sx, sy)
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis("off")
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation="nearest")
    plt.savefig(filename)
    plt.close()
    # plt.show()


def newlayer(layer, g):
    newlayer = copy.deepcopy(layer)
    try:
        newlayer.weight = nn.Parameter(g(newlayer.weight))
    except AttributeError:
        pass

    try:
        newlayer.bias = nn.Parameter(g(newlayer.bias))
    except AttributeError:
        pass

    return newlayer


class LRP(object):
    """Our Layer-wise relevance propagation (LRP Class).
    It seems to me we make our prediction and pass it back
    through each layer until we reach the input, at which point
    we might have a gradient. The original is in numpy so we
    convert everything into numpy arrays.
    """

    def __init__(self, model, points, obj, normaliser, layerid=1, num_points=350, sigma=2.0, device="cpu"):
        super(LRP, self).__init__()
        self.model = model
        self.points = points
        self.obj = obj
        self.device = device
        self.batch_size = self.model._final.size()[0]
        self.normaliser = normaliser
        self.layerid = layerid
        self.num_data_points = num_points
        self.sigma = sigma

    def _gen_rot(self, rx, ry, rz):
        """Return a transformation with rotations in radians"""
        rx = torch.tensor([rx])
        ry = torch.tensor([ry])
        rz = torch.tensor([rz])
        axis = VecRotTen(rx, ry, rz)
        # axis = axis.to(self.device) # onto the GPU potentially
        return axis

    def _perform_lrp(self, result, target):
        """Attempt the heatmapping.org/tutorial code instead. This seems faster
        and a little more elegant, plus it actually appears to work. Not sure
        why it differs from the above but I suspect it's all in the rollaxis
        stuff I did which is probably wrong. Ultimately, this is using the
        forward and backward functions per layer in reverse to do the LRP."""

        self.model.eval()
        layers = (
            [self.model._modules["conv1"]]
            + [self.model._modules["conv2"]]
            + [self.model._modules["conv2b"]]
            + [self.model._modules["conv3"]]
            + [self.model._modules["conv3b"]]
            + [self.model._modules["conv4"]]
            + [self.model._modules["conv4b"]]
            + [self.model._modules["conv5"]]
            + [self.model._modules["conv5b"]]
            + [self.model._modules["conv6"]]
            + [toconv(self.model._modules["fc1"])]
            + [toconv(self.model._modules["fc2"])]
        )
        L = len(layers)
        # print(layers)

        A = []
        A.append(target)
        R = []

        for _ in range(0, L):
            R.append(None)

        for idx in range(L):
            # Because we have a 'view' in our net (from conv to linear) we add a reshape here
            if idx == 10:
                A[idx] = A[idx].reshape(-1, num_flat_features(A[idx]), 1, 1)

            A.append(layers[idx].forward(A[idx]))

        R.append(A[-1].data)

        for l in range(1, L)[::-1]:
            print("Layer accumulating:", l)
            # opposite of our reshape above
            if l == 9:
                # TODO base this on next layer bit
                R[l + 1] = R[l + 1].reshape(-1, 256, 2, 2)

            A[l] = A[l].data
            A[l].requires_grad_(requires_grad=True)

            if isinstance(layers[l], torch.nn.MaxPool2d):
                layers[l] = torch.nn.AvgPool2d(2)

            if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

                # if l <= 1:      rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
                # if 5 <= l <= 2: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
                # if l >= 6:      rho = lambda p: p;                       incr = lambda z: z+1e-9
                def rho(p):
                    return p

                def incr(z):
                    return z + 1e-9

                z = incr(newlayer(layers[l], rho).forward(A[l]))  # step 1
                s = (R[l + 1] / z).data
                (z * s).sum().backward()
                c = A[l].grad  # step 3
                R[l] = (A[l] * c).data  # step 4

            else:
                print("Skip layer")
                R[l] = R[l + 1]

        # Apparently, the last pixel layer needs different rules
        A[0] = (A[0].data).requires_grad_(True)

        lb = (A[0].data * 0 + 0).requires_grad_(True)
        hb = (A[0].data * 0 + 1).requires_grad_(True)

        # step 1 (a)
        z = layers[0].forward(A[0]) + 1e-9
        z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
        # step 2
        s = (R[1] / z).data
        (z * s).sum().backward()
        c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
        # step 4
        R[0] = (A[0] * c + lb * cp + hb * cm).data

        return R

    def run(self):
        device = "cpu"

        # Ensure an equal spread of data around all the rotation space so
        # we don't miss any particular areas
        twopie = math.pi * 2.0
        pp = twopie / self.num_data_points
        rx = 0
        ry = 0
        rz = 0
        tx = torch.tensor([0], dtype=torch.float32, device=self.device)
        ty = torch.tensor([0], dtype=torch.float32, device=self.device)

        dps = []

        for i in range(self.num_data_points):
            # tx = (random.random() - 0.5) * trans_scale
            # ty = (random.random() - 0.5) * trans_scale
            trans = TransTen(tx, ty)
            rot = self._gen_rot(rx, ry, rz)
            dps.append((rot, trans))
            rx += pp
            ry += pp
            rz += pp

        rx = 0
        ry = 0
        rz = 0
        tx = 0
        ty = 0
        rot = self._gen_rot(rx, ry, rz)

        for i in range(self.num_data_points):
            # tx = (random.random() - 0.5) * trans_scale
            # ty = (random.random() - 0.5) * trans_scale
            trans = TransTen(
                torch.tensor([math.sin(tx)], dtype=torch.float32, device=self.device),
                torch.tensor([math.cos(ty)], dtype=torch.float32, device=self.device),
            )
            dps.append([rot, trans])
            tx += pp
            ty += pp

        splat = Splat(device=device)
        splat.grads = False
        loaded_points = load_obj(objpath=self.obj)
        scaled_points = PointsTen(device)
        scaled_points.from_points(loaded_points)

        mask = []
        for _ in loaded_points:
            mask.append(1.0)
        mask = torch.tensor(mask, device=device)

        save_dir = "./lrp_anim"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # we need to update older models with a few parameters I think
        if not hasattr(self.model.splat, "grads"):
            self.model.splat.grads = False

        for idx, (r, t) in enumerate(dps):
            # Setup our splatting pipeline which is added to both dataloader
            # and our network as they use the same settings
            result = splat.render(scaled_points, r, t, mask, sigma=self.sigma)
            # trans_points = splat.transform_points(scaled_points, xr, yr, zr, xt, yt)
            path = save_dir + "/in_" + str(idx).zfill(3) + ".jpg"
            save_image(result.clone().cpu(), path)
            target = result.reshape(1, 128, 128)
            target = self.normaliser.normalise(target.repeat(self.batch_size, 1, 1, 1))
            R = self._perform_lrp(self.model.get_render_params(), target)
            B = self.model.forward(target, self.points)
            # self._perform_lrp(self.model.get_render_params(), target, scaled_points, sigma)
            # for i,l in enumerate([4,3,2,1,0]):
            #    heatmap(np.array(R[l][0]).sum(axis=0),0.5*i+1.5,0.5*i+1.5, "heatmap.png")

            #path = save_dir + "/heat_0_" + str(idx).zfill(3) + ".jpg"
            #heatmap(np.array(R[0][0]).sum(axis=0), 3.5, 3.5, path)
            #path = save_dir + "/heat_1_" + str(idx).zfill(3) + ".jpg"
            #heatmap(np.array(R[1][0]).sum(axis=0), 3.0, 3.0, path)
            #path = save_dir + "/heat_2_" + str(idx).zfill(3) + ".jpg"
            #heatmap(np.array(R[2][0]).sum(axis=0), 2.5, 2.5, path)
            #path = save_dir + "/heat_3_" + str(idx).zfill(3) + ".jpg"
            #heatmap(np.array(R[3][0]).sum(axis=0), 2.0, 2.0, path)
            path = save_dir + "/heat_" + str(self.layerid) + "_" + str(idx).zfill(3) + ".jpg"
            heatmap(np.array(R[self.layerid][0]).sum(axis=0), 1.0, 1.0, path)
            path = save_dir + "/out_" + str(idx).zfill(3) + ".jpg"
            save_image(self.normaliser.normalise(B).cpu()[0].reshape(128, 128), path)

            del B
            del R
            del target
            del result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lrp for shaper")
    parser.add_argument(
        "--savedir", default="./save", help="Path to our saved model (default: ./save)."
    )
    parser.add_argument(
        "--obj",
        default="teapot.obj",
        help="The obj file for this network (default: teapot.obj).",
    )
    parser.add_argument(
        "--layerid", type=int, default=1, help="The layer to generate the heatmap for(default: 1)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="The input sigma (default: 2.0).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=350,
        help="The Number of Points (default: 350).",
    )

    device = "cpu"
    args = parser.parse_args()
    savename = "checkpoint.pth.tar"

    model = load_model(args.savedir + "/model.tar", device)

    (model, points, _, _, _, _, prev_args) = load_checkpoint(
        model, args.savedir, savename, device
    )

    normaliser = NormaliseNull()
    if prev_args.normalise_basic:
        normaliser = NormaliseBasic()

    model.to(device)
    lrp = LRP(model, points, args.obj, normaliser, args.layerid, args.num_points, args.sigma)
    lrp.run()
