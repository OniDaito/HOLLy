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
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import os
import sys
from util.points import load_points, save_points, init_points
from util.loadsave import save_checkpoint, save_model
from data.loader import Loader
from data.imageload import ImageLoader
from data.sets import DataSet, SetType
from data.buffer import Buffer, BufferImage
from data.batcher import Batcher
from stats import stats as S
from net.renderer import Splat
from net.net import Net
from util.image import NormaliseNull, NormaliseTorch
from util.math import PointsTen


def calculate_loss(target: torch.Tensor, output: torch.Tensor):
    """
    Our loss function, used in train and test functions.

    Parameters
    ----------

    target : torch.Tensor
        The target, properly shaped.

    output : torch.Tensor
        The tensor predicted by the network, not shaped

    Returns
    -------
    Loss
        A loss object
    """

    loss = F.l1_loss(output, target, reduction="sum")
    return loss


def test(
    args,
    model,
    buffer_test: Buffer,
    epoch: int,
    step: int,
    points: PointsTen,
    sigma: float,
    write_fits=False,
):
    """
    Switch to test / eval mode and do some recording to our stats
    program and see how we go.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    model : torch.nn.Module
        The main net model
    buffer_test : Buffer
        The buffer that represents our test data.
    epoch : int
        The current epoch.
    step : int
        The current step.
    points : PointsTen
        The current PointsTen being trained.
    sigma : float
        The current sigma.
    write_fits : bool
        Write the intermediate fits files for analysis.
        Takes up a lot more space. Default - False.
    Returns
    -------
    None
    """

    # Put model in eval mode
    model.eval()

    # Which normalisation are we using?
    normaliser = NormaliseNull()

    if args.normalise_basic:
        normaliser = NormaliseTorch()

    image_choice = random.randrange(0, args.batch_size)
    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_test, batch_size=args.batch_size)
    rots_in = []  # Save rots in for stats
    rots_out = []  # Collect all rotations out

    if args.objpath != "":
        # Assume we are simulating so we have rots to save
        S.watch(rots_in, "rotations_in_test")
        S.watch(rots_out, "rotations_out_test")

    for batch_idx, ddata in enumerate(batcher):
        # turn off grads because for some reason, memory goes BOOM!
        with torch.no_grad():
            # Offsets is essentially empty for the test buffer.
            target = ddata[0]
            target_shaped = normaliser.normalise(
                target.reshape(args.batch_size, 1, args.image_size, args.image_size)
            )

            output = normaliser.normalise(model(target_shaped, points))
            output = output.reshape(
                args.batch_size, 1, args.image_size, args.image_size
            )

            rots_out.append(model.get_rots())
            loss = calculate_loss(args, target_shaped, output)

            # Just save one image for now - first in the batch
            if batch_idx == image_choice:
                target = torch.squeeze(target_shaped[0])
                output = torch.squeeze(output[0])
                S.save_jpg(target, args.savedir, "in_e", epoch, step, batch_idx)
                S.save_jpg(output, args.savedir, "out_e", epoch, step, batch_idx)
                S.save_fits(target, args.savedir, "in_e", epoch, step, batch_idx)
                S.save_fits(output, args.savedir, "out_e", epoch, step, batch_idx)

                if write_fits:
                    S.write_immediate(target, "target_image", epoch, step, batch_idx)
                    S.write_immediate(output, "output_image", epoch, step, batch_idx)

                if args.predict_sigma:
                    ps = model._final.shape[1] - 1
                    sp = nn.Softplus(threshold=12)
                    sig_out = torch.tensor([torch.clamp(sp(x[ps]), max=14) for x in model._final])
                    S.watch(sig_out, "sigma_out_test")

            # soft_plus = torch.nn.Softplus()
            S.watch(loss, "loss_test")  # loss saved for the last batch only.
            if args.objpath != "":
                # Assume we are simulating so we have rots to save
                rots_in.append(ddata[1])

    buffer_test.set.shuffle()
    model.train()


def cont_sigma(args, current_epoch: int, sigma: float, sigma_lookup: list) -> float:
    """
    If we are using _cont_sigma, we need to work out the linear
    relationship between the points. We call this each step.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    current_epoch : int
        The current epoch.
    sigma : float
        The current sigma.
    sigma_lookup : list
        The sigma lookup list of floats.

    Returns
    -------
    float
        The sigma to use
    """

    eps = args.epochs / (len(sigma_lookup) - 1)
    a = 0
    if current_epoch > 0:
        a = int(current_epoch / eps)
    b = a + 1
    assert b < len(sigma_lookup)
    assert a >= 0

    ssig = sigma_lookup[a]
    esig = sigma_lookup[b]

    ssize = args.train_size
    # TODO - we really should use the loader size here
    if args.aug:
        ssize = args.train_size * args.num_aug
    steps_per_epoch = ssize / args.batch_size
    steps = steps_per_epoch * eps
    cont_factor = math.pow(float(esig) / float(ssig), 1.0 / float(steps))
    new_sigma = sigma * cont_factor
    return new_sigma


def train(
    args,
    device,
    sigma_lookup,
    model,
    points,
    buffer_train,
    buffer_test,
    data_loader,
    optimiser,
):
    """
    Now we've had some setup, lets do the actual training.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    device : str
        The device to run the model on (cuda / cpu)
    sigma_lookup : list
        The list of float values for the sigma value.
    model : nn.Module
        Our network we want to train.
    points : PointsTen
        The points we want to sort out.
    buffer_train :  Buffer
        The buffer in front of our training data.
    data_loader : Loader
        A data loader (image or simulated).
    optimiser : torch.optim.Optimizer
        The optimiser we want to use.

    Returns
    -------
    None
    """

    model.train()

    # Which normalisation are we using?
    normaliser = NormaliseNull()

    if args.normalise_basic:
        normaliser = NormaliseTorch()

    sigma = sigma_lookup[0]

    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_train, batch_size=args.batch_size)

    # Begin the epochs and training
    for epoch in range(args.epochs):
        if not args.cont:
            sigma = sigma_lookup[min(epoch, len(sigma_lookup) - 1)]

        # Set the sigma - two seems too many
        model.set_sigma(sigma)
        data_loader.set_sigma(sigma)

        # Now begin proper
        print("Starting Epoch", epoch)
        for batch_idx, ddata in enumerate(batcher):
            target = ddata[0]
            optimiser.zero_grad()

            # Shape and normalise the input batch
            target_shaped = normaliser.normalise(
                target.reshape(args.batch_size, 1, args.image_size, args.image_size)
            )

            output = normaliser.normalise(model(target_shaped, points))

            loss = calculate_loss(target_shaped, output)
            loss.backward()
            lossy = loss.item()
            optimiser.step()

            # If we are using continuous sigma, lets update it here
            if args.cont and not args.no_sigma:
                sigma = cont_sigma(args, epoch, sigma, sigma_lookup)
                # 2 places again - not ideal :/
                data_loader.set_sigma(sigma)
                model.set_sigma(sigma)

            # We save here because we want our first step to be untrained
            # network
            if batch_idx % args.log_interval == 0:
                # Add watches here
                S.watch(lossy, "loss_train")
                # Temporary ignore of images in the DB
                # S.watch(target[0], "target")
                # S.watch(output[0], "output")
                if args.predict_sigma:
                    S.watch(sigma, "sigma_in")

                print(
                    "Train Epoch: \
                    {} [{}/{} ({:.0f}%)]\tLoss Main: {:.6f}".format(
                        epoch,
                        batch_idx * args.batch_size,
                        buffer_train.set.size,
                        100.0 * batch_idx * args.batch_size / buffer_train.set.size,
                        lossy
                    )
                )

                if args.save_stats:
                    test(args, model, buffer_test, epoch, batch_idx, points, sigma)
                    S.save_points(points, args.savedir, epoch, batch_idx)
                    S.update(epoch, buffer_train.set.size, args.batch_size, batch_idx)

            if batch_idx % args.save_interval == 0:
                print("saving checkpoint", batch_idx, epoch)
                save_model(model, args.savedir + "/model.tar")

                save_checkpoint(
                    model,
                    points,
                    optimiser,
                    epoch,
                    batch_idx,
                    loss,
                    sigma,
                    args,
                    args.savedir,
                    args.savename,
                )

        buffer_train.set.shuffle()

    # Save a final points file once training is complete
    S.save_points(points, args.savedir, epoch, batch_idx)
    return points


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
    # valid_set_size = args.valid_size
    test_set_size = args.test_size

    if args.aug:
        train_set_size = args.train_size * args.num_aug
        # valid_set_size = args.valid_size * args.num_aug
        test_set_size = args.test_size * args.num_aug

    # Sigma checks. Do we use a file, do we go continuous etc?
    # Check for sigma blur file
    sigma_lookup = [None]

    if not args.no_sigma:
        sigma_lookup = [10.0, 1.25]
        if len(args.sigma_file) > 0:
            if os.path.isfile(args.sigma_file):
                with open(args.sigma_file, "r") as f:
                    ss = f.read()
                    sigma_lookup = []
                    tokens = ss.replace("\n", "").split(",")
                    for token in tokens:
                        sigma_lookup.append(float(token))

    if (args.no_sigma and not args.predict_sigma) is True:
        print("If using no-sigma, you must predict sigma")
        sys.exit()

    # Setup our splatting pipeline. We use two splats with the same
    # values because one never changes its points / mask so it sits on
    # the gpu whereas the dataloader splat reads in differing numbers of
    # points.

    splat_in = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device, size=(args.image_size, args.image_size))
    splat_out = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device, size=(args.image_size, args.image_size))

    # Setup the dataloader - either generated from OBJ or fits
    if args.fitspath != "":
        data_loader = ImageLoader(
            size=args.train_size + args.test_size,
            image_path=args.fitspath,
            sigma=sigma_lookup[0],
        )

        set_train = DataSet(
            SetType.TRAIN, train_set_size, data_loader, alloc_csv=args.allocfile
        )
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)

        buffer_train = BufferImage(
            set_train, buffer_size=args.buffer_size, device=device,
                image_size=(args.image_size, args.image_size)
        )
        buffer_test = BufferImage(set_test, buffer_size=test_set_size, 
            image_size=(args.image_size, args.image_size),  device=device)

    elif args.objpath != "":
        data_loader = Loader(
            size=args.train_size + args.test_size,
            objpath=args.objpath,
            wobble=args.wobble,
            dropout=args.dropout,
            spawn=args.spawn_rate,
            max_spawn=args.max_spawn,
            translate=(not args.no_data_translate),
            sigma=sigma_lookup[0],
            max_trans=args.max_trans,
            augment=args.aug,
            num_augment=args.num_aug
        )

        fsize = min(data_loader.size - test_set_size, train_set_size)
        set_train = DataSet(SetType.TRAIN, fsize, data_loader, alloc_csv=args.allocfile)
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)

        buffer_train = Buffer(
            set_train, splat_in, buffer_size=args.buffer_size, device=device
        )

        buffer_test = Buffer(
            set_test, splat_in, buffer_size=test_set_size, device=device
        )
    else:
        raise ValueError("You must provide either fitspath or objpath argument.")

    # TODO - possibly remove fast-forward and what not.
    # TODO - Loading for retraining should go somewhere else. We hardly ever
    # do that these days anyway

    points = init_points(
        args.num_points, device=device, deterministic=args.deterministic
    )
    model = Net(
        splat_out,
        predict_translate=(not args.no_translate),
        predict_sigma=args.predict_sigma,
        max_trans=args.max_trans
    ).to(device)

    # Load our init points as well, if we are loading the same data
    # file later on - this is only in initialisation
    if os.path.isfile(args.savedir + "/points.csv"):
        print("Loading points file", args.savedir + "/points.csv")
        tpoints = load_points(args.savedir + "/points.csv")
        points = PointsTen(device=device)
        points.from_points(tpoints)
    else:
        points = init_points(num_points=args.num_points, device=device)
        save_points(args.savedir + "/points.csv", points)

    points.data.requires_grad_(requires_grad=True)

    # Save the training data to disk so we can interrogate it later
    if args.save_train_data:
        set_train.save(args.savedir + "/train_set.pickle")
        data_loader.save(args.savedir + "/train_data.pickle")

    variables = []
    variables.append({"params": model.parameters()})
    optimiser = optim.AdamW(variables, lr=args.lr)
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
        data_loader,
        optimiser
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
        "--save-train-data",
        action="store_true",
        default=False,
        help="Save the data used in the training (default: False).",
    )
    parser.add_argument(
        "--predict-sigma",
        action="store_true",
        default=False,
        help="Predict the sigma (default: False).",
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
        "--no-translate",
        action="store_true",
        default=False,
        help="Turn off translation prediction in the network \
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
        "--cont",
        default=False,
        action="store_true",
        help="Continuous sigma values",
        required=False,
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
        "--no-sigma",
        default=False,
        action="store_true",
        help="Do we use an input sigma profile or do we ignore it?",
        required=False,
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=10,
        help="how many augmentations to perform per datum (default 10)",
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
        "--image-size",
        type=int,
        default=128,
        help="The size of the images involved, assuming square \
                          (default: 128).",
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
