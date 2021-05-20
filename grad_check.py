""" 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/  
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

grad_check.py - An attempt to look at the gradients
and see if we are suffering from the shattered
gradient problem.

http://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf

"""

import pickle
import copy
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from util.loadsave import load_checkpoint, load_model
from util.plyobj import load_obj
from net.renderer import Splat
from util.math import PointsTen, VecRotTen, TransTen


class GradCheck(object):
    """We plot the points based on nearby images to check if we are
    getting a shattered gradients problem.
    """

    def __init__(self, model, points, obj, device):
        super(GradCheck, self).__init__()
        self.model = model
        self.points = points
        self.obj = obj
        self.fc2_grads = []
        self.fc2_activations = []
        self.device = device

    def add_grad_forward(self):
        """The forward hook saves the output at each layer so we
        can use it when we do the back-prop."""

        def grad_hook(module, input, output):
            print("Setting forward hook.")
            module.saved_output = output.cpu().detach().numpy()
            module.saved_input = input[0].cpu().detach().numpy()

        return grad_hook

    def _gen_rot(self, rx, ry, rz):
        """Return a transformation with rotations in radians"""
        rx = torch.tensor([rx])
        ry = torch.tensor([ry])
        rz = torch.tensor([rz])
        axis = VecRotTen(rx, ry, rz)
        # axis = axis.to(self.device) # onto the GPU potentially
        return axis

    def plot_heatmap_grads(self, filename):
        dframe_grads = pd.DataFrame()
        for idx, grads in enumerate(self.fc2_grads):
            # (512 * 5 so quite a bit)
            grad = np.array(grads.cpu().flatten().numpy(), dtype=np.float)
            dframe_grads["rot" + str(idx).zfill(3)] = grad

        sns.heatmap(dframe_grads)
        plt.savefig(filename)
        plt.close()

    def plot_heatmap_activations(self, filename):
        dframe_active = pd.DataFrame()

        for idx, actives in enumerate(self.fc2_activations):
            # ( Just 5 I think so not so many)
            active = np.array(actives.flatten(), dtype=np.float)
            dframe_active["rot" + str(idx).zfill(3)] = active

        sns.heatmap(dframe_active)
        plt.savefig(filename)
        plt.close()

    def process_results(self):
        self.plot_heatmap_grads("heatmap_grads.png")
        self.plot_heatmap_activations("heatmap_activations.png")

    def run(self):
        import os.path

        if os.path.isfile("grad.pickle"):
            with open("grad.pickle", "rb") as f:
                (self.fc2_activations, self.fc2_grads) = pickle.load(f)
                self.process_results()
            return

        # Ensure an equal spread of data around all the rotation space so
        # we don't miss any particular areas
        num_data_points = 360
        sigma = 2.8
        batch_size = self.model._final.size()[0]

        twopie = math.pi * 2.0
        pp = twopie / num_data_points  # ** (1. / 3)
        rx = 0
        ry = 0
        rz = 0
        tx = 0
        ty = 0

        dps = []

        for i in range(num_data_points):
            # tx = (random.random() - 0.5) * trans_scale
            # ty = (random.random() - 0.5) * trans_scale
            rot = self._gen_rot(rx, ry, rz)
            dps.append([rot.x, rot.y, rot.z, tx, ty])
            rz += pp
            ry += pp
            rz += pp

            # if rz > twopie:
            #    rz = 0.0
            #    ry += pp
            #    if ry > twopie:
            #        ry = 0.0
            #        rx += pp
            #        if rx > twopie:
            #            rx = 0.0

        self.model.eval()
        self.model.fc2.register_forward_hook(self.add_grad_forward())

        # Setup our splatting pipeline which is added to both dataloader
        # and our network as they use the same settings
        splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=self.device)
        loaded_points = load_obj(objpath=self.obj)
        scaled_points = PointsTen(device=self.device)
        scaled_points.from_points(loaded_points)

        mask = []
        for _ in loaded_points:
            mask.append(1.0)
        mask = torch.tensor(mask, device=device)

        # we need to update older models with a few parameters I think
        if not hasattr(self.model.splat, "grads"):
            self.model.splat.grads = True

        self.model.to(device)

        for dp in dps:
            xr = torch.tensor([dp[0]], dtype=torch.float32, device=self.device)
            yr = torch.tensor([dp[1]], dtype=torch.float32, device=self.device)
            zr = torch.tensor([dp[2]], dtype=torch.float32, device=self.device)
            xt = torch.tensor([dp[3]], dtype=torch.float32, device=self.device)
            yt = torch.tensor([dp[4]], dtype=torch.float32, device=self.device)

            r = VecRotTen(xr, yr, zr)
            t = TransTen(xt, yt)

            result = splat.render(scaled_points, r, t, mask, sigma=sigma)
            # trans_points = splat.transform_points(scaled_points, xr, yr, zr, xt, yt)
            # save_image(result.clone().cpu(), "lrp_in.jpg")

            target = result.reshape(1, 128, 128)
            target = target.repeat(batch_size, 1, 1, 1)
            target = target.to(device)
            points.data.requires_grad_(requires_grad=True)
            # We use tpoints because otherwise we can't update points
            # and keep working out the gradient cos pytorch weirdness
            output = model.forward(target, points)
            output = output.reshape(batch_size, 1, 128, 128)
            loss = F.l1_loss(output, target)
            print("Loss:", loss)
            loss.backward()
            # print("Rotations returned:", model.get_rots())

            with torch.no_grad():
                # print("Grads:", model.conv1.grad)
                # print("Grads:", points.grad.shape)
                # print("Grads2:", model.conv1.weight.grad.shape)
                # print("Grads3:", model.fc2.weight.grad) # 5, 512
                self.fc2_grads.append(copy.deepcopy(self.model.fc2.weight.grad))
                self.fc2_activations.append(copy.deepcopy(self.model.fc2.saved_output))

                self.model.conv1.weight.grad.zero_()
                self.model.conv2.weight.grad.zero_()
                self.model.conv3.weight.grad.zero_()
                self.model.conv4.weight.grad.zero_()
                self.model.conv5.weight.grad.zero_()
                self.model.conv6.weight.grad.zero_()
                self.model.fc1.weight.grad.zero_()
                self.model.fc2.weight.grad.zero_()
                # points.data.grad.zero_() # this should be possible I think

            # loss.backward() # this sets up all the LRP calls
            # output = torch.squeeze(output.cpu()[0])
            # save_image(output, "lrp_out.jpg")
            # Now perform the LRP algorithm
            # self._perform_check(model.get_rots(), target)
        with open("grad.pickle", "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(
                (self.fc2_activations, self.fc2_grads), f, pickle.HIGHEST_PROTOCOL
            )

        self.process_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Check for shaper.")
    parser.add_argument(
        "--savedir", default="./save", help="Path to our saved model (default: ./save)."
    )
    parser.add_argument(
        "--obj",
        default="teapot.obj",
        help="The obj file for this network (default: teapot.obj).",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA."
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    savename = "checkpoint.pth.tar"
    (model, points) = load_checkpoint(args.savedir, savename, device, evaluation=True)
    model = load_model(args.savedir + "/model.tar", device)
    gcx = GradCheck(model, points, args.obj, device)
    gcx.run()
