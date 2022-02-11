""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

run.py - an attempt to find the 3D shape from an image.

To load a trained network:
  python run.py --load checkpoint.pth.tar --image <X> --points <Y>

  Load a checkpoint and an image (FITS) and output an
  image and some angles.

  For example:

  python run.py --load ../runs/2021_05_06_bl_0 --image renderer.fits --points ../runs/2021_05_06_bl_0/last.ply

"""

import torch
import math
import argparse
import sys
import os
from net.renderer import Splat
from util.image import save_image, load_fits, save_fits
from util.loadsave import load_checkpoint, load_model
from util.plyobj import load_obj, load_ply
from util.math import PointsTen
import torch.nn.functional as F
from util.image import NormaliseBasic, NormaliseNull
from PIL import Image


def _print_rotations(self, input, output):
    """Internal function that is attached via a hook when we
    want to see the rotations from the fc3 layer.
    """
    # output is a Tensor. output.data is the Tensor we are interested
    print("Inside " + self.__class__.__name__ + " forward")
    print("")
    print("input: ", type(input))
    print("input[0]: ", type(input[0]))
    print("output: ", type(output))
    print("")
    print("input size:", input[0].size())
    print("output size:", output.data.size())
    print("output norm:", output.data.norm())

    # Here we set our rotations globally which is a bit naughty
    # but for now it's ok. Callbacks need to have something passed.


def file_test(model, device, sigma, input_image):
    """Test our model by making an image, printing the rotations
    and then seeing what our model comes up with.
    """
    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    with torch.no_grad():
        model.eval()
        model.to(device)

        im = Image.open(args.imagefile)
        if im.size != (128, 128):
            print("Input image is not equal to 128,128")
            sys.exit()
        fm = torch.zeros(
            im.size, dtype=torch.float32, requires_grad=False, device=device
        )

        for y in range(0, im.size[1]):
            for x in range(0, im.size[0]):
                fm[y][x] = im.getpixel((x, y)) / 255.0

        fm = fm.reshape((1, 1, 128, 128))
        fm.to(device)
        model.set_sigma(sigma)
        x = model.forward(fm, points)
        # print("Output rotations:", grx, gry, grz)
        # im = gen_baseline(grx, gry, grz, "output.bmp", objpath = args.obj)


def image_test(model, points, device, sigma, input_image, normaliser):
    """Test our model by loading an image and seeing how well we
    can match it. We might need to duplicate to match the batch size.
    """
    # splat_in = Splat(device=device)
    splat_out = Splat(device=device)
    model.set_splat(splat_out)

    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    with torch.no_grad():
        model.eval()
        im = normaliser.normalise(input_image.reshape((1, 1, 128, 128)))
        im = im.to(device)
        model.set_sigma(sigma)
        x = normaliser.normalise(model.forward(im, points))
        x = torch.squeeze(x)
        im = torch.squeeze(im)
        loss = F.l1_loss(x, im, reduction="sum")
        print(float(loss.item()), ",", model.get_rots())
        save_image(x, name="guess.jpg")

        if os.path.exists("guess.fits"):
            os.remove("guess.fits")

        save_fits(x, name="guess.fits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shaper run")
    parser.add_argument("--load", default=".", help="Path to our model dir.")
    parser.add_argument(
        "--image", default="input.fits", help="An input image in FITS format"
    )
    parser.add_argument(
        "--points", default="", help="Alternative points to use (default: none)."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.2, help="Sigma for the output (default: 1.2)"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.load and os.path.isfile(args.load + "/checkpoint.pth.tar"):
        # (savedir, savename) = os.path.split(args.load)
        # print(savedir, savename)
        model = load_model(args.load + "/model.tar")
        (model, points, _, _, _, _, prev_args) = load_checkpoint(
            model, args.load, "checkpoint.pth.tar", device
        )
        model = model.to(device)
        model.eval()
    else:
        print("--load must point to a run directory.")
        sys.exit(0)

    points = PointsTen(device=device)

    normaliser = NormaliseNull()
    if prev_args.normalise_basic:
        normaliser = NormaliseBasic()

    # Potentially load a different set of points
    if args.points != "":
        if "ply" in args.points:
            points.from_points(load_ply(args.points))
        else:
            points.from_points(load_obj(args.points))

    if os.path.isfile(args.image):
        input_image = load_fits(args.image, flip=True)
        image_test(model, points, device, args.sigma, input_image, normaliser)
    else:
        print("--image must point to a valid fits file.")
        sys.exit(0)
