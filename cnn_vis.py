"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

cnn_vis.py - an attempt to visualise the layers of a CNN.

A good explanation of this can be found at:
https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

Essentially, we take a random image and run it through the network, using the
gradient at that layer we are interested in as the target, moving the image
towards it.

Based on the work by:
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
import copy
import argparse
import math
import torch
from torch.autograd import Variable
from torch.optim import Adam
from util.loadsave import load_model, load_checkpoint
from util.image import save_image
from net.net import num_flat_features
from net.renderer import Splat
from util.plyobj import load_obj
from util.math import TransTen, VecRotTen, PointsTen
import torch.nn.functional as F


def process_finals(finals, savedir):
    """ Take all our layers and their filters and create a set of images. """
    from matplotlib.colors import hsv_to_rgb

    maxi = -1e10
    mini = 1e10

    for layer in finals:
        for fylter in layer:
            tmax = np.amax(fylter)
            if tmax > maxi:
                maxi = tmax
            tmin = np.amin(fylter)
            if tmin < mini:
                mini = tmin

    spread = maxi
    if math.fabs(mini) > spread:
        spread = math.fabs(mini)

    for lidx, layer in enumerate(finals):
        for fidx, fylter in enumerate(layer):
            final_image = np.zeros([128, 128, 3], dtype=np.uint8)
            red_image = copy.copy(fylter)
            blue_image = copy.copy(fylter)
            red_image[red_image > 0] = 0
            red_image *= -1
            blue_image[blue_image < 0] = 0

            hsv_blue = np.zeros([128, 128, 3], dtype=np.float32)
            hsv_blue[:, :, 0] = 240 / 360
            hsv_blue[:, :, 1] = 1.0  # blue_image[0] / spread
            hsv_blue[:, :, 2] = blue_image[0] / spread

            hsv_red = np.zeros([128, 128, 3], dtype=np.float32)
            hsv_red[:, :, 0] = 0
            hsv_red[:, :, 1] = 1.0  # red_image[0] / spread
            hsv_red[:, :, 2] = red_image[0] / spread

            final_image = np.round(hsv_to_rgb(hsv_blue) + hsv_to_rgb(hsv_red) * 255)

            # created_image = recreate_image(vimage)
            im_path = (
                savedir + "/layer_vis_l" + str(lidx + 1) + "_f" + str(fidx) + ".png"
            )
            save_image(final_image, name=im_path)


class CNNLayerVisualization:
    """
    Produces an image that minimizes the loss of a convolution
    operation for a specific layer and filter
    """

    def __init__(
        self,
        model,
        image_size: int,
        savedir: str,
        selected_layer,
        obj_path: str,
        selected_filter: int,
        device,
    ):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.image_size = image_size
        self.save_dir = savedir
        self.device = device
        self.obj = obj_path

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        x = self.selected_layer
        x.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, initial_image):
        # Hook the selected layer
        self.hook_layer()
        # initial_image = initial_image.unsqueeze(0)
        # initial_image = initial_image.unsqueeze(0)

        random_image = torch.Tensor(
            np.float32(np.random.uniform(0, 1.0, (1, 128, 128)))
        )
        random_image = random_image.unsqueeze(0)
        vimage = random_image.to(self.device)

        ze = torch.tensor([0.001], dtype=torch.float32, device=device)
        # Try an actual image instead
        rot = VecRotTen(ze, ze, ze)
        rot.randomise()
        trans = TransTen(ze, ze)

        splat = Splat(device=device)
        splat.grads = False
        loaded_points = load_obj(objpath=self.obj)
        scaled_points = PointsTen(device)
        scaled_points.from_points(loaded_points)
        mask = []

        for _ in loaded_points:
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        vimage = splat.render(scaled_points, rot, trans, mask, sigma=2.8)
        vimage = vimage.unsqueeze(0).unsqueeze(0)

        # vimage = Variable(random_image, requires_grad=True)
        # vimage = vimage.to(self.device)

        # Define optimizer for the image
        # TODO - why these settings?
        optimizer = Adam([vimage], lr=0.0001, weight_decay=1e-6)
        for i in range(1, 31):  # TODO - why 31 here? Num steps?
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = vimage

            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to
                # trigger
                # the forward hook function
                if isinstance(layer, torch.nn.Linear):
                    x = x.view(-1, num_flat_features(x))
                # We use leak_relu's in our model so reflect that here.
                x = F.leaky_relu(layer(x))

                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected
            # layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            # print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(
            #    loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # Save image
            # if i == 30:

            # created_image = recreate_image(vimage)
            # im_path = self.save_dir + '/layer_vis_l' + \
            #    str(self.selected_layer) + \
            #    '_f' + str(self.selected_filter) + '_iter' + \
            #    str(i) + '.jpg'
            # save_image(created_image, name=im_path)

            return vimage.cpu().data.numpy()[0]


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="PyTorch CNN Vis")
    parser.add_argument("--savedir", help="The path to saved run")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--savename",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="./objs/bunny_large.obj",
        help="Obj for generated image (default: ./objs/bunny_large.obj)",
    )
    parser.add_argument(
        "--filter", type=int, default=5, help="Which filter to use (default: 5)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="The size of the images involved, assuming square \
                          (default: 128).",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if os.path.isfile(args.savedir + "/" + args.savename):
        (model, points) = load_checkpoint(
            args.savedir, args.savename, device, evaluation=True
        )
        model = load_model(args.savedir + "/model.tar", device)
        model.to(device)
        # print("Loaded model", model)
        layers = []
        for layer in model:
            print("Layer", layer)
            if isinstance(layer, torch.nn.Conv2d):
                filters = []
                for f in range(layer.out_channels):
                    layer_vis = CNNLayerVisualization(
                        model, args.image_size, args.savedir, layer, args.obj, f, device
                    )
                    final_image = layer_vis.visualise_layer_with_hooks(model)
                    filters.append(final_image)
                layers.append(filters)
            print("Finished Layer", layer)

        process_finals(layers, args.savedir)

        # Layer visualization without pytorch hooks
        # layer_vis.visualise_layer_without_hooks()
