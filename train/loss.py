""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

loss.py - loss functions used in training

"""
import torch
from util.math import PointsTen
from train.icp_test import rmsd_score
import torch.nn.functional as F


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


def calculate_move_loss(prev_points: PointsTen, new_points: PointsTen):
    """
    How correlated is our movement from one step to the next? Use
    ICP and Nearest Neighbour RMSD

    Parameters
    ----------

    prev_points : PointsTen
        The starting points

    new_points : PointsTen
        The points as updated by the network

    Returns
    -------
    Loss : float
        The loss score
    """
    p0 = prev_points.get_points()
    pv0 = []
    for p in p0.data:
        pv0.append((p.x, p.y, p.z))

    p1 = new_points.get_points()
    pv1 = []
    for p in p1.data:
        pv1.append((p.x, p.y, p.z))

    loss = rmsd_score(pv0, pv1)
    return loss
