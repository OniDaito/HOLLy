""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

angle_vis.py - Visualise the angles from our random
rotation generation.

"""

from pyquaternion import Quaternion
from net.renderer import Splat
import torch.nn.functional as F
from util.plyobj import load_obj
from util.loadsave import load_checkpoint, load_model
import torch
from random import random
import numpy as np
import argparse
import math
import pickle
import os
from tqdm import tqdm
import scipy.stats
from util.math import VecRotTen, VecRot, TransTen, PointsTen, qdist, vec_to_quat, angles_to_axis
from util.image import NormaliseTorch, NormaliseNull


SCALE = 40
TITLE = "Visualising rotations."


def basic_viz(rot_pairs):
    """
    Given input and output rotations plot in a way
    that is easy to visualise.

    Parameters
    ----------
    rot_pairs : List of Tuple of VecRot
        List of input/output rotation pairs

    Returns
    -------
    self
    """
    data_matrix = np.zeros([SCALE, SCALE, SCALE], dtype=np.uint8)
    count_matrix = np.zeros([SCALE, SCALE, SCALE], dtype=np.uint8)

    for pair in rot_pairs:
        rot = pair[0]
        q = Quaternion(axis=rot.get_normalised(),
                        radians=rot.get_length())
        rot_f = VecRot(q.axis[0] * q.radians,
                            q.axis[1] * q.radians,
                            q.axis[2] * q.radians)

        x = int(rot_f.x * 5 + SCALE / 2)
        y = int(rot_f.y * 5 + SCALE / 2)
        z = int(rot_f.z * 5 + SCALE / 2)

        # Now get the error at this spot
        q0 = vec_to_quat(rot)
        q1 = vec_to_quat(pair[1])
        dd = qdist(q0, q1)

        data_matrix[x, y, z] += dd
        count_matrix[x, y, x] += 1

    count_matrix = np.where(count_matrix < 1.0, 1, count_matrix)
    data_matrix = data_matrix / count_matrix
    rot_max = np.max(data_matrix)

    from vedo import Volume, show

    vol = Volume(data_matrix, c='RdBu_r', alpha=0.1, mode=1)
    vol.addScalarBar3D()

    #lego = vol.legosurface(vmin=1, vmax=rot_max)
    #lego.shrink()
    #lego.addScalarBar3D()

    show(vol, TITLE, axes=1).close()


def sigma_effect(args, model, points, prev_args, device):
    """
    What effect does sigma have on the loss, particularly
    with different rotations.
    Do rotations that differ a lot give a bigger error or no?

    Parameters
    ----------
    args : namespace
        The program command line arguments    
    
    model : Net
        Our neural network model
    
    points: Points
        The points the model came up with

    prev_args : dictionary
        The arguments used by the network when it was run.
    
    device : 
        The torch device we are running on.

    Returns
    -------
    None
    """
    import pprint

    dim_size = args.dim_size # how many angles to compare to each other
    sigmas = [10,9.0,8.1,7.29,6.56,5.9,5.31,4.78,4.3,3.87,3.65,3.28,2.95,2.66,2.39,2.15,1.94,1.743,1.57,1.41]
    # Which normalisation are we using?
    normaliser = NormaliseNull()

    if prev_args.normalise_basic:
        normaliser = NormaliseTorch()

    mask = []
    for _ in range(len(points)):
        mask.append(1.0)
    
    mask = torch.tensor(mask, device=device)
    base_points = PointsTen(device=device)
    base_points.from_points(load_obj(args.obj))
    mask_base = []
    
    for _ in range(len(base_points)):
        mask_base.append(1.0)

    mask_base = torch.tensor(mask_base, device=device)
    xt = torch.tensor([0.0], dtype=torch.float32)
    yt = torch.tensor([0.0], dtype=torch.float32)
    t = TransTen(xt, yt)

    # TODO - maybe a pandas dataframe is ideal here?

    # Build our cube of results
    # Each entry has the two angles and the error
    error_cube = []
    for s in sigmas:
        xlist = []

        for x in range(dim_size):
            ylist = []
            rx = VecRot(0, 0, 0).random().to_ten(device=device)

            for y in range(dim_size):
                ry = VecRot(0, 0, 0).random().to_ten(device=device)
                if x > 0:
                    ry = xlist[0][y][1]

                # Rotation 0, Rotation 1, Rotation network, qdist, loss, loss network 
                q0 = vec_to_quat(rx)
                q1 = vec_to_quat(ry)
                rdist = qdist(q0, q1)
                ylist.append([rx, ry, 0, rdist, 0, 0])

            xlist.append(ylist)
        error_cube.append(xlist)

    splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)

    for sidx in tqdm(range(len(sigmas))):
        current_sigma = sigmas[sidx]

        xlist = error_cube[sidx]

        for xidx in range(dim_size-1):
            r0 = error_cube[sidx][xidx][0][0]

            base_image = splat.render(base_points, r0, t, mask_base, sigma=current_sigma)
            base_image = base_image.reshape(1, 1, 128, 128)
            base_image = normaliser.normalise(base_image)

            model_image = model.forward(base_image, points)
            model_image = normaliser.normalise(model_image.reshape(1, 1, 128, 128))
            loss_model = F.l1_loss(model_image, base_image)
            model_image = torch.squeeze(model_image.cpu()[0])
            model_rots = model.get_rots()
                
            for yidx in range(xidx+1, dim_size):
                r1 = error_cube[sidx][xidx][yidx][1]

                second_image = splat.render(base_points, r1, t, mask_base, sigma=current_sigma)
                second_image = second_image.reshape(1, 1, 128, 128)
                second_image = normaliser.normalise(second_image)
                second_image = second_image.squeeze()

                base_image = base_image.squeeze()

                loss_base = F.l1_loss(base_image, second_image)
                rdist = error_cube[sidx][xidx][yidx][3]
                error_cube[sidx][xidx][yidx][2] = model_rots
                error_cube[sidx][xidx][yidx][4] = loss_base.item()
                error_cube[sidx][xidx][yidx][5] = loss_model.item()

                error_cube[sidx][yidx][xidx][2] = model_rots
                error_cube[sidx][yidx][xidx][4] = loss_base.item()
                error_cube[sidx][yidx][xidx][5] = loss_model.item()

                #print("Sigma, Dist, Loss", current_sigma, rdist, loss.item())

    # Now see if there are any correlations?
    # Start with the distances

    print("Correlations between Distance and error per sigma")

    for sidx in range(len(sigmas)):
        dists = []
        losses = []
        losses_network = []
        losses_network.append(error_cube[sidx][dim_size-1][0][5])

        for x in range(dim_size-1):
            losses_network.append(error_cube[sidx][x][0][5])

            for y in range(x, dim_size):
                dists.append(error_cube[sidx][x][y][3])
                losses.append(error_cube[sidx][x][y][4])

        print("Sigma Results:", sigmas[sidx])
        print("------------------")
        pp = pprint.PrettyPrinter(indent=4, width=dim_size * 10)
        pp.pprint(error_cube[sidx]
        #r = np.corrcoef(dists, losses)
        t = scipy.stats.kendalltau(dists, losses)
        #print("Correlation Pearsons", r)
        print("Correlation Tau", t)
        #r = np.corrcoef(dists, losses)
        t = scipy.stats.kendalltau(dists, losses_network)
        #print("Correlation Pearsons", r)
        print("Correlation Tau Model", t)

    print("Correlations between Sigma and error")
    print("------------------------------------")

    fsigs = []
    losses = []
    losses_model = []

    for sidx in range(len(sigmas)):
        for x in range(dim_size-1):
            losses_model.append(error_cube[sidx][x][0][5])

            for y in range(x, dim_size):
                fsigs.append(sigmas[sidx])
                losses.append(error_cube[sidx][x][y][4])

    r = np.corrcoef(fsigs, losses)
    t = scipy.stats.kendalltau(fsigs, losses)
    #print("Correlation Pearsons Base", r)
    print("Correlation Tau Base", t)

    r = np.corrcoef(fsigs, losses_model)
    t = scipy.stats.kendalltau(fsigs, losses_model)
    #print("Correlation Pearsons Model", r)
    print("Correlation Tau Model", t)

    print("Correlation between Sigma and Variance on the loss")
    fsigs = []
    variances = []
    variances_model = []

    for sidx in range(len(sigmas)):
        fsigs.append(sigmas[sidx])
        losses = []
        losses_model = []
        losses_model.append(error_cube[sidx][dim_size-1][0][5])

        for x in range(dim_size):
            for y in range(dim_size):
                if y != x:
                    losses.append(error_cube[sidx][x][y][4])
                    losses_model.append(error_cube[sidx][x][y][5])

        variances.append(np.var(losses))
        variances_model.append(np.var(losses_model))

    r = np.corrcoef(fsigs, variances)
    t = scipy.stats.kendalltau(fsigs, variances)
    #print("Correlation Pearson Base", r)
    print("Correlation Tau Base", t)
    print("Variances Base:", variances)

    r = np.corrcoef(fsigs, variances_model)
    t = scipy.stats.kendalltau(fsigs, variances_model)
    #print("Correlation Pearsons Model", r)
    print("Correlation Tau Model", t)
    print("Variances Model:", variances_model)


def angle_check(args, model, points, prev_args, device):
    """
    Given our model and some input angles, run through the 
    network and see what corresponding angles we get.

    Parameters
    ----------
    args : namespace
        The program command line arguments    
    
    model : Net
        Our neural network model
    
    points: Points
        The points the model came up with

    prev_args : dictionary
        The arguments used by the network when it was run.
    
    device : 
        The torch device we are running on.

    Returns
    -------
    List
        a List of tuples of VecRot pairs
    """

    xt = 0.0
    yt = 0.0
    xt = torch.tensor([xt], dtype=torch.float32, device=device)
    yt = torch.tensor([yt], dtype=torch.float32, device=device)
    trans = TransTen(xt, yt)

    normaliser = NormaliseNull()
    if prev_args.normalise_basic:
        normaliser = NormaliseTorch()

    # Load some base points from an obj
    loaded_points = load_obj(objpath=args.obj)
    mask = []
    for _ in loaded_points:
        mask.append(1.0)
    mask = torch.tensor(mask, device=device)
    scaled_points = PointsTen(device=device).from_points(loaded_points)
    model.set_sigma(args.sigma)

    rots_in_out = []

    for i in range(args.num_rots):
        rot = VecRot(0, 0, 0)
        rot.random()
        rot = rot.to_ten(device=device)
        splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)
        target = splat.render(scaled_points, rot, trans, mask, sigma=args.sigma)
        target = target.reshape(1, 128, 128)
        target = target.repeat(prev_args.batch_size, 1, 1, 1)
        target = target.to(device)
        target = normaliser.normalise(target)
        output = model.forward(target, points)
        output = normaliser.normalise(output.reshape(prev_args.batch_size, 1, 128, 128))
        loss = F.l1_loss(output, target)
        prots = model.get_rots().squeeze()
        print("Loss:", loss.item())
        rots_in_out.append((rot, VecRot(float(prots[0][0]), float(prots[0][1]), float(prots[0][2]))))
        del target
        del output

    return rots_in_out

def load(args, device):
    """ Begin our training routine on the selected device."""
    # Continue training or start anew
    # Declare the variables we absolutely need
    model = None
    points = None
    model = load_model(args.savedir + "/model.tar", device)

    if os.path.isfile(args.savedir + "/" + args.savename):
        (model, points, _, _, _, _, prev_args) = load_checkpoint(
            model, args.savedir, args.savename, device
        )
        model.to(device)
        print("Loaded model", model)
    else:
        print("Error - need to pass in a model")
        return

    model.eval()

    with torch.no_grad():
        # results = angle_check(args, model, points, prev_args, device)
        sigma_effect(args, model, points, prev_args, device)

    #basic_viz(results)


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Shaper Eval")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument(
        "--obj", default="teapot.obj", help="Path to the groundruth obj file"
    )
    parser.add_argument(
        "--sigma", default=1.25, type=float, help="The sigma value for this testing"
    )
    parser.add_argument(
        "--num-rots", default=360, type=int, help="The number of rots to try (default 360)."
    )
    parser.add_argument(
        "--dim-size", default=20, type=int, help="How many angles in loss check (default 20)."
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
  
    parser.add_argument(
        "--rots",
        metavar="R",
        type=float,
        nargs=3,
        help="Rotation around X, Y, Z axis in degrees.",
    )
    parser.add_argument(
        "--trans", metavar="R", type=float, nargs=2, help="Translation on X, Y plane"
    )
    parser.add_argument(
        "--savename",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )
   
    # Initial setup of PyTorch
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    load(args, device)
    print("Finished Angle Vis")
