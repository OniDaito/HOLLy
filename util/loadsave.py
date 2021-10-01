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


def save_checkpoint(
    model, points, optimiser, epoch, batch_idx, loss, sigma, args, savedir, savename
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
    sigma : float
        The current sigma
    args : args object
        The args object this model was running with
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
            "sigma": sigma,
            "args": args,
            "optimiser_state_dict": optimiser.state_dict(),
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
    model, savedir, savename, device="cpu"
):
    """Load our checkpoint, given the full path to the checkpoint.
    A model must be loaded and passed in already. We set the parameters
    of this model from these stored in the checkpoint.

    Parameters
    ----------
    model : Model
        A model created blank or loaded with load_model
    savedir : str
        The directory of the save files
    savename : str
        The name of the save file
    device : 
        We must pass the device in
    Returns
    -------
    tuple

    """

    checkpoint = torch.load(savedir + "/" + savename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # this line seems to fail things :/
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    batch_idx = checkpoint["batch_idx"]
    args = checkpoint["args"]
    points = checkpoint["points"]
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    points = points.data.to(device)
    model = model.to(device)
    model.predict_sigma = args.predict_sigma
    model.predict_translate = not args.no_translate
    model.max_trans = args.max_trans

    return (model, points, optimiser, epoch, batch_idx, loss, args)


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
