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
from util.math import VecRotTen, VecRot, TransTen, PointsTen
from util.image import NormaliseTorch, NormaliseNull


SCALE = 40
NUM_ITEMS = 400
TITLE = "Visualising rotations."

def dotty(p, q):
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]


def qdist(q0, q1):
    q0_minus_q1 = [q0[0] - q1[0], q0[1] - q1[1], q0[2] - q1[2], q0[3] - q1[3]]
    d_minus = math.sqrt(dotty(q0_minus_q1, q0_minus_q1))
    q0_plus_q1 = [q0[0] + q1[0], q0[1] + q1[1], q0[2] + q1[2], q0[3] + q1[3]]
    d_plus = math.sqrt(dotty(q0_plus_q1, q0_plus_q1))
    if d_minus < d_plus:
        return d_minus
    return d_plus


def qrotdiff(q0, q1):
    d = dotty(q0, q1)
    d = math.fabs(d) 
    return 2.0 * math.acos(d)


def vec_to_quat(rv):
    angle = math.sqrt(rv.x * rv.x + rv.y * rv.y + rv.z * rv.z)
    ax = rv.x / angle
    ay = rv.x / angle
    az = rv.x / angle

    qx = ax * math.sin(angle/2)
    qy = ay * math.sin(angle/2)
    qz = az * math.sin(angle/2)
    qw = math.cos(angle/2)
    return (qx, qy, qz, qw)


def basic_viz(rot_pairs):
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

    count_matrix = [1 if i < 1 else i for i in count_matrix]
    data_matrix = data_matrix / count_matrix
    rot_max = np.max(data_matrix)

    from vedo import Volume, show

    vol = Volume(data_matrix, c='RdBu_r', alpha=0.1, mode=1)
    vol.addScalarBar3D()

    lego = vol.legosurface(vmin=1, vmax=rot_max)
    lego.shrink()
    lego.addScalarBar3D()

    show(vol, TITLE, axes=1).close()


def angle_check(args, model, points, prev_args, device):
    xt = 0.0
    yt = 0.0
    xt = torch.tensor([xt], dtype=torch.float32, device=device)
    yt = torch.tensor([yt], dtype=torch.float32, device=device)
    trans = TransTen(xt, yt)

    #normaliser = NormaliseNull()
    #if prev_args.normalise_basic:
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

    for i in range(NUM_ITEMS):
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


def load_pickled(args):
    results = pickle.load(open('angle_vis.pickle', 'rb'))
    basic_viz(results)


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
        results = angle_check(args, model, points, prev_args, device)

    pickle.dump(results, open('angle_vis.pickle', 'wb'))
    basic_viz(results)


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
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--pickled", action="store_true", default=False, help="Load a pickle file and skip the net part (default False)"
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
    if args.pickled:
        load_pickled(args)
    else:
        load(args, device)
    print("Finished Angle Vis")
    sys.exit(0)
