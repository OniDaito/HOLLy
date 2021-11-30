""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

loss.py - given two images, calculate the loss

To load a trained network:
  python loss.py --i <path to image> --j <path to image>

"""

import torch
import math
import argparse
import sys
import os
from util.image import load_fits
from util.image import NormaliseTorch, NormaliseNull
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOLLy loss")

    parser.add_argument(
        "--i", default="input.fits", help="An input image in FITS format"
    )
    parser.add_argument(
        "--j", default="input2.fits", help="An input image in FITS format"
    )
    parser.add_argument(
        "--norm",
        default=False,
        action="store_true",
        help="Normalise with the basic normaliser.",
        required=False,
    )

    args = parser.parse_args()
    normaliser = NormaliseNull()

    if args.norm:
        normaliser = NormaliseTorch()

    # Potentially load a different set of points
    if os.path.isfile(args.i) and os.path.isfile(args.j):
        i_image = load_fits(args.i, flip=True)
        j_image = load_fits(args.j, flip=True)
        loss = F.l1_loss(i_image, j_image, reduction="mean")
        print(float(loss.item()))

    else:
        print("--i and --j must point to a valid fits files.")
        sys.exit(0)
