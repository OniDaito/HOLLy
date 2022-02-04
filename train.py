""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train.py - an attempt to find the 3D shape from an image.
To train a network, use:
  python train.py <OPTIONS>

See the README file and the __main__ function for the
various options.

"""

import torch
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import os
import sys
from util.points import init_points_poisson, load_points, save_points, init_points, init_points_spot
from util.loadsave import save_checkpoint, save_model
from data.loader import Loader
from data.imageload import ImageLoader
from data.sets import DataSet, SetType
from data.buffer import Buffer, BufferImage
from stats import stats as S
from net.renderer import Splat
from net.net import Net
from util.math import PointsTen
from train.train import train


def init(args, device):
    """
    Initialise all of our models, optimizers and other useful
    things before passing on to train.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    device : str
        The device to run the model on (cuda / cpu)

    Returns
    -------
    None
    """

    # Continue training or start anew
    # Declare the variables we absolutely need
    model = None
    points = None
    buffer_train = None
    buffer_test = None
    data_loader = None
    optimiser = None

    train_set_size = args.train_size
    valid_set_size = args.valid_size
    test_set_size = args.test_size

    if args.aug:
        train_set_size = args.train_size * 4
        valid_set_size = args.valid_size * 4
        test_set_size = args.test_size * 4
    # Sigma checks. Do we use a file, do we go continuous etc?
    # Check for sigma blur file

    sigma_lookup = [10.0, 1.25]
    if len(args.sigma_file) > 0:
        if os.path.isfile(args.sigma_file):
            with open(args.sigma_file, "r") as f:
                ss = f.read()
                sigma_lookup = []
                tokens = ss.replace("\n", "").split(",")
                for token in tokens:
                    sigma_lookup.append(float(token))

    # Setup our splatting pipeline. We use two splats with the same
    # values because one never changes its points / mask so it sits on
    # the gpu whereas the dataloader splat reads in differing numbers of
    # points.

    splat_in = Splat(
        math.radians(90),
        1.0,
        1.0,
        10.0,
        device=device,
        size=(args.image_height, args.image_width),
    )
    splat_out = Splat(
        math.radians(90),
        1.0,
        1.0,
        10.0,
        device=device,
        size=(args.image_height, args.image_width),
    )

    # Setup the dataloader - either generated from OBJ or fits
    if args.fitspath != "":
        data_loader = ImageLoader(
            size=args.train_size + args.test_size + args.valid_size,
            image_path=args.fitspath,
            sigma=sigma_lookup[0],
        )

        set_train = DataSet(
            SetType.TRAIN, train_set_size, data_loader, alloc_csv=args.allocfile
        )
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)
        set_validate = DataSet(SetType.VALID, valid_set_size, data_loader)

        buffer_train = BufferImage(
            set_train,
            buffer_size=args.buffer_size,
            device=device,
            image_size=(args.image_height, args.image_width),
        )
        buffer_test = BufferImage(
            set_test,
            buffer_size=test_set_size,
            image_size=(args.image_height, args.image_width),
            device=device,
        )
        buffer_valid = BufferImage(
            set_validate,
            buffer_size=valid_set_size,
            image_size=(args.image_height, args.image_width),
            device=device,
        )

    elif args.objpath != "":
        data_loader = Loader(
            size=args.train_size + args.test_size + args.valid_size,
            objpath=args.objpath,
            wobble=args.wobble,
            dropout=args.dropout,
            spawn=args.spawn_rate,
            max_spawn=args.max_spawn,
            translate=(not args.no_data_translate),
            sigma=sigma_lookup[0],
            max_trans=args.max_trans,
            augment=args.aug,
            num_augment=4,
        )

        fsize = min(data_loader.size - test_set_size, train_set_size)
        set_train = DataSet(SetType.TRAIN, fsize, data_loader, alloc_csv=args.allocfile)
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)
        set_validate = DataSet(SetType.VALID, valid_set_size, data_loader)

        buffer_train = Buffer(
            set_train, splat_in, buffer_size=args.buffer_size, device=device
        )

        buffer_test = Buffer(
            set_test, splat_in, buffer_size=test_set_size, device=device
        )

        buffer_valid = Buffer(
            set_validate, splat_in, buffer_size=valid_set_size, device=device
        )
    else:
        raise ValueError("You must provide either fitspath or objpath argument.")

    # TODO - possibly remove fast-forward and what not.
    # TODO - Loading for retraining should go somewhere else. We hardly ever
    # do that these days anyway

    points = init_points(
        args.num_points, device=device, deterministic=args.deterministic
    )

    if args.ipspot:
        points = init_points_spot(
            args.num_points, device=device, deterministic=args.deterministic
        )
    elif args.poisson:
        points = init_points_poisson(
            args.num_points, device=device
        )

    model = Net(
        splat_out,
        max_trans=args.max_trans,
    ).to(device)

    if args.poseonly:
        from util.plyobj import load_obj, load_ply

        if "obj" in args.objpath:
            points = load_obj(objpath=args.objpath)
        elif "ply" in args.objpath:
            points = load_ply(args.objpath)
        else:
            raise ValueError("If using poseonly, objpath must be set.")

        points = PointsTen(device=device).from_points(points)

    else:
        # Load our init points as well, if we are loading the same data
        # file later on - this is only in initialisation
        if os.path.isfile(args.savedir + "/points.csv"):
            print("Loading points file", args.savedir + "/points.csv")
            tpoints = load_points(args.savedir + "/points.csv")
            points = PointsTen(device=device)
            points.from_points(tpoints)
        else:
            save_points(args.savedir + "/points.csv", points)

        points.data.requires_grad_(requires_grad=True)

    # Save the training and test data to disk so we can interrogate it later
    set_train.save(args.savedir + "/train_set.pickle")
    set_test.save(args.savedir + "/test_set.pickle")
    data_loader.save(args.savedir + "/train_data.pickle")

    variables = []
    variables.append({"params": model.parameters(), "lr": args.lr})

    if not args.poseonly:
        variables.append({"params": points.data, "lr": args.plr})

    optimiser = optim.AdamW(variables)

    print("Starting new model")

    # Now start the training proper
    train(
        args,
        device,
        sigma_lookup,
        model,
        points,
        buffer_train,
        buffer_test,
        buffer_valid,
        data_loader,
        optimiser,
    )

    save_model(model, args.savedir + "/model.tar")


if __name__ == "__main__":
    # Training settings
    # TODO - potentially too many options now so go with a conf file?
    parser = argparse.ArgumentParser(description="PyTorch Shaper Train")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="input batch size for training \
                          (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.004,
        help="learning rate (default: 0.004)",
    )
    parser.add_argument(
        "--plr",
        type=float,
        default=0.0004,
        help="learning rate (default: 0.0004)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=1.0,
        help="Probabilty of spawning a point \
                          (default: 1.0).",
    )
    parser.add_argument(
        "--max-trans",
        type=float,
        default=0.1,
        help="The scalar on the translation we generate and predict \
                          (default: 0.1).",
    )
    parser.add_argument(
        "--reduction",
        type=float,
        default=0.75,
        help="The factor to reduce the learning rate by \
                          (default: 0.75).",
    )
    parser.add_argument(
        "--max-spawn",
        type=int,
        default=1,
        help="How many flurophores are spawned total. \
                          (default: 1).",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        default=False,
        help="Save the stats of the training for later \
                          graphing.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically",
    )
    parser.add_argument(
        "--ipspot",
        action="store_true",
        default=False,
        help="Initialise points across space with a gaussian and a spot in the middle.\
                        (default: false)",
    )
    parser.add_argument(
        "--poisson",
        action="store_true",
        default=False,
        help="Initialise points across space with a gaussian and a spot in the middle.\
                        (default: false)",
    )
    parser.add_argument(
        "--no-data-translate",
        action="store_true",
        default=False,
        help="Turn off translation in the data \
                            loader(default: false)",
    )
    parser.add_argument(
        "--normalise-basic",
        action="store_true",
        default=False,
        help="Normalise with torch basic intensity divide",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training \
                          status",
    )
    parser.add_argument(
        "--pinterval",
        type=int,
        default=1000,
        metavar="N",
        help="how many steps to wait before checking the points learning rate (default: 1000)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=200,
        help="how many points to optimise (default 200)",
    )
    parser.add_argument(
        "--aug",
        default=False,
        action="store_true",
        help="Do we augment the data with XY rotation (default False)?",
        required=False,
    )
    parser.add_argument(
        "--poseonly",
        default=False,
        action="store_true",
        help="Only optimise the pose. Default false",
        required=False,
    )
    parser.add_argument(
        "--adapt",
        default=False,
        action="store_true",
        help="Adaptive learning rate (default: False)",
        required=False,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="The width of the input and output images \
                          (default: 320).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=150,
        help="The height of the input and output images \
                          (default: 150).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="how many batches to wait before saving.",
    )
    parser.add_argument(
        "--load",
        help="A checkpoint file to load in order to continue \
                          training",
    )
    parser.add_argument(
        "--savename",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )
    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument(
        "--allocfile", default=None, help="An optional data order allocation file."
    )
    parser.add_argument(
        "--sigma-file", default="", help="Optional file for the sigma blur dropoff."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="When coupled with objpath, what is the chance of \
                          a point being dropped? (default 0.0)",
    )
    parser.add_argument(
        "--wobble",
        type=float,
        default=0.0,
        help="Distance to wobble our fluorophores \
                          (default 0.0)",
    )
    parser.add_argument(
        "--fitspath",
        default="",
        help="Path to a directory of FITS files.",
        required=False,
    )
    parser.add_argument(
        "--objpath",
        default="",
        help="Path to the obj for generating data",
        required=False,
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=50000,
        help="The size of the training set (default: 50000)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="The size of the training set (default: 200)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=200,
        help="The size of the training set (default: 200)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=40000,
        help="How big is the buffer in images? \
                          (default: 40000)",
    )
    args = parser.parse_args()

    # Stats turn on
    if args.save_stats:
        S.on(args.savedir)

    # Initial setup of PyTorch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print("Using device", device)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    init(args, device)
    print("Finished Training")
    S.close()
