""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train.py - the main network training routines.

"""
import torch
import torch.optim as optim
import math
from data.loader import Loader
from data.buffer import Buffer, BaseBuffer
from data.batcher import Batcher
from util.math import PointsTen
from util.image import NormaliseNull, NormaliseBasic
from train.loss import calculate_loss, calculate_move_loss
import numpy as np
from util.loadsave import save_checkpoint, save_model
from stats import stats as S
from net.net import Net
from util.math import PointsTen
from train.test import test


def cont_sigma(
    args, current_epoch: int, batch_idx: int, batches_epoch: int, sigma_lookup: list
) -> float:
    """
    Using _cont_sigma, we need to work out the linear
    relationship between the points. We call this each step.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    current_epoch : int
        The current epoch.
    batch_idx : int
        The current batch number
    batches_epoch : int
        The number of batches per epoch
    sigma_lookup : list
        The sigma lookup list of floats.

    Returns
    -------
    float
        The sigma to use
    """
    progress = float(current_epoch * batches_epoch + batch_idx) / float(
        args.epochs * batches_epoch
    )
    middle = (len(sigma_lookup) - 1) * progress
    start = int(math.floor(middle))
    end = int(math.ceil(middle))
    between = middle - start
    s_sigma = sigma_lookup[start]
    e_sigma = sigma_lookup[end]
    new_sigma = s_sigma + ((e_sigma - s_sigma) * between)

    return new_sigma


def validate(
    args,
    model,
    buffer_valid: Buffer,
    points: PointsTen,
):
    """
    Switch to test / eval mode and run a validation step.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    model : torch.nn.Module
        The main net model
    buffer_valid: Buffer
        The buffer that represents our validation data.
    points : PointsTen
        The current PointsTen being trained.
    sigma : float
        The current sigma.

    Returns
    -------
    Loss
        A float representing the validation loss
    """
    # Put model in eval mode
    model.eval()

    # Which normalisation are we using?
    normaliser = NormaliseNull()

    if args.normalise_basic:
        normaliser = NormaliseBasic()

    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_valid, batch_size=args.batch_size)
    ddata = batcher.__next__()

    with torch.no_grad():
        # Offsets is essentially empty for the test buffer.
        target = ddata[0]
        target_shaped = normaliser.normalise(
            target.reshape(args.batch_size, 1, args.image_height, args.image_width)
        )

        output = normaliser.normalise(model(target_shaped, points))
        output = output.reshape(args.batch_size, 1, args.image_height, args.image_width)
        valid_loss = calculate_loss(target_shaped, output).item()

    buffer_valid.set.shuffle()
    model.train()
    return valid_loss


def train(
    args,
    device,
    sigma_lookup,
    model: Net,
    points: PointsTen,
    buffer_train: BaseBuffer,
    buffer_test: BaseBuffer,
    buffer_validate: BaseBuffer,
    data_loader: Loader,
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
    optimiser_points : torch.optim.Optimizer
        The optimiser for the points
    Returns
    -------
    None
    """

    model.train()

    # Set a lower limit on the lr, with a lower one on the plr. Factor is less harsh.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        patience=int(args.epochs / 2),
        factor=args.reduction,
        min_lr=[args.lr / 10, args.plr / 100],
    )

    # Which normalisation are we using?
    normaliser = NormaliseNull()

    if args.normalise_basic:
        normaliser = NormaliseBasic()

    sigma = sigma_lookup[0]
    data_loader.set_sigma(sigma)
    S.watch(optimiser.param_groups[0]["lr"], "learning_rate")

    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_train, batch_size=args.batch_size)

    # Keep a copy of the points so we can test for a good structure
    prev_points = points.clone()

    # Begin the epochs and training
    for epoch in range(args.epochs):

        # Now begin proper
        print("Starting Epoch", epoch)
        for batch_idx, ddata in enumerate(batcher):
            target = ddata.data
            optimiser.zero_grad()

            # Shape and normalise the input batch
            target_shaped = normaliser.normalise(
                target.reshape(args.batch_size, 1, args.image_height, args.image_width)
            )

            output = normaliser.normalise(model(target_shaped, points))
            loss = calculate_loss(target_shaped, output)
            loss.backward()
            lossy = loss.item()
            optimiser.step()
            sigma = cont_sigma(args, epoch, batch_idx, len(batcher), sigma_lookup)
            data_loader.set_sigma(sigma)

            # We save here because we want our first step to be untrained
            # network
            if batch_idx % args.log_interval == 0:
                # Add watches here
                S.watch(lossy, "loss_train")
                S.watch(sigma, "sigma_in")

                # Watch the training rotations too!
                if args.objpath != "" and args.save_stats:
                    S.watch(ddata.rotations, "rotations_in_train")
                    S.watch(model.get_rots(), "rotations_out_train")

                print(
                    "Train Epoch: \
                    {} [{}/{} ({:.0f}%)]\tLoss Main: {:.6f}".format(
                        epoch,
                        batch_idx * args.batch_size,
                        buffer_train.set.size,
                        100.0 * batch_idx * args.batch_size / buffer_train.set.size,
                        lossy,
                    )
                )

                if args.save_stats:
                    test(args, model, buffer_test, epoch, batch_idx, points, sigma)
                    S.save_points(points, args.savedir, epoch, batch_idx)
                    S.update(epoch, buffer_train.set.size, args.batch_size, batch_idx)

            steps = batch_idx + (epoch * (buffer_train.set.size / args.batch_size))

            if steps % args.pinterval == 0 and args.adapt:
                # Now attempt to see if we have a good model
                # Calculate the move loss and adjust the learning rate on the points accordingly
                # We need a window of at least 10 steps at log interval 100.
                move_loss = calculate_move_loss(prev_points, points)
                scheduler.step(move_loss)
                S.watch(move_loss, "move_loss")
                new_plr = optimiser.param_groups[1]["lr"]
                print(
                    "Learning Rates (pose, points):",
                    str(optimiser.param_groups[0]["lr"]),
                    str(optimiser.param_groups[1]["lr"]),
                )
                S.watch(new_plr, "points_lr")
                prev_points = points.clone()

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
