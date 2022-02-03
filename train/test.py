""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

test.py - testing our network during training

"""

import torch
import torch.nn as nn
import random
from data.buffer import Buffer
from data.batcher import Batcher
from util.math import PointsTen
from util.image import NormaliseNull, NormaliseTorch
from stats import stats as S
from train.loss import calculate_loss


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
        if args.altloss:
            normaliser.factor = 1000.0

    image_choice = random.randrange(0, args.batch_size)
    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_test, batch_size=args.batch_size)
    rots_in = []  # Save rots in for stats
    rots_out = []  # Collect all rotations out
    test_loss = 0

    if args.objpath != "" and args.save_stats:
        # Assume we are simulating so we have rots to save
        S.watch(rots_in, "rotations_in_test")
        S.watch(rots_out, "rotations_out_test")

    for batch_idx, ddata in enumerate(batcher):
        # turn off grads because for some reason, memory goes BOOM!
        with torch.no_grad():
            # Offsets is essentially empty for the test buffer.
            target = ddata.data
            target_shaped = normaliser.normalise(
                target.reshape(args.batch_size, 1, args.image_height, args.image_width)
            )

            output = normaliser.normalise(model(target_shaped, points))
            output = output.reshape(
                args.batch_size, 1, args.image_height, args.image_width
            )

            rots_out.append(model.get_rots())
            test_loss += calculate_loss(target_shaped, output).item()

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

                ps = model._final.shape[1] - 1
                sp = nn.Softplus(threshold=12)
                sig_out = torch.tensor(
                    [torch.clamp(sp(x[ps]), max=14) for x in model._final]
                )
                S.watch(sig_out, "sigma_out_test")

            # soft_plus = torch.nn.Softplus()
            if args.objpath != "":
                # Assume we are simulating so we have rots to save
                rots_in.append(ddata.rotations)

    test_loss /= len(batcher)
    S.watch(test_loss, "loss_test")  # loss saved for the last batch only.
    buffer_test.set.shuffle()
    model.train()
