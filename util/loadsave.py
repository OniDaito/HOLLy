""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/        # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/        # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_loadsave.py - load and save our model. Also
deal with checkpoints.

"""

import torch
import torch.optim as optim
import math
from net.net import Net
from net.renderer import Splat


def save_checkpoint(
    model, points, optimiser, epoch, batch_idx, loss, savedir, savename
):
    """
    Saving a checkpoint out along with optimizer and other useful
    values. Points aren't quite part of the model yet so we add them
    too.

    Parameters
    ----------
    model : NN.module
        The model
    points : torch.tensor
        The points the network has derived
    optimiser :
        The optimiser used by the training function.
    epoch : int
        The epoch we have reached
    batch_idx : int
        The batch_idx we got to during training
    loss:
        The current loss
    savedir : str
        The path to save to
    savename: str
        The name of the save file

    Returns
    -------
    None

    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "points": points,
            "batch_idx": batch_idx,
            "optimizer_state_dict": optimiser.state_dict(),
            "loss": loss,
        },
        savedir + "/" + savename,
    )


def save_model(model, path):
    """
    Save the model

    Parameters
    ----------
    model : NN.module
        The model
    path : str
        The path (including file name)

    Returns
    -------
    None

    """
    torch.save(model, path)


def load_checkpoint(
    savedir, savename, device, lr=0.0004, evaluation=False, predict_sigma=False
):
    """Load our checkpoint, given the full path to the checkpoint.
    We can load for eval or continue training, so sometimes we ignore
    the optimizer.

    Parameters
    ----------
    savedir : str
        The directory of the save files
    savename : str
        The name of the save file
    device : str
        CUDA or cpu?
    lr : float
        The learning rate (default 0.0004)
    evaluation: book
        Load in evaluation mode (default False)
    predict_sigma: bool
        Does this model predict sigma

    Returns
    -------
    None

    """

    splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)
    model = Net(splat, predict_sigma=predict_sigma)
    checkpoint = torch.load(savedir + "/" + savename, map_location=device)
    if hasattr(checkpoint, "model_state_dict"):
        model.load_state_dict(checkpoint["model_state_dict"])
    elif hasattr(checkpoint, "model_main_state_dict"):
        # older versions had model_main
        model.load_state_dict(checkpoint["model_main_state_dict"])

    points = checkpoint["points"]
    points = points.data.to(device)

    model = model.to(device)

    if evaluation is True:
        return (model, points)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # this line seems to fail things :/
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    batch_idx = checkpoint["batch_idx"]

    return (model, points, optimizer, epoch, batch_idx, loss)


def load_model(path, device="cpu"):
    """
    Load the model

    Parameters
    ----------

    path : str
        The path (including file name)

    device: str
        The device we are using, CUDA or cpu

    Returns
    -------
    None

    """
    return torch.load(path, map_location=device)
